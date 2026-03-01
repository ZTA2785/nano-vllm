import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        # 先设置 config 值
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # 启动进程组并绑定 CUDA 设备、设置 torch 数据类型
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        # 加载模型和 sampler
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        # 预热，避免首次真实推理时出现不确定的延迟或内存不足的问题
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        # 设置默认设备和 dtype
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 如果采用了张量并行，那么就使用共享内存来实现不同进程之间的通讯，在 read_shm 和 write_shm 中有使用，并且 rank=0 的进程负责创建，其他进程直接连接
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier() # 先创建再等待其他进程同步
            else:
                dist.barrier() # 先同步得到 rank=0 进程创建的共享内存，然后再连接
                self.shm = SharedMemory(name="nanovllm")
                self.loop() # 其他进程进入 loop 开始循环，等待 rank=0 进程通过共享内存发送指令

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    # 通过共享内存（shared memory, shm）和事件（event）机制，实现不同进程之间的远程方法调用。
    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    # 统一接口 —— 如果当前进程是主进程（rank == 0），那么在调用本地方法之前，会先把请求广播到共享内存中，让所有 worker 一起执行；如果是 worker，则直接调用本地方法。
    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    # 预热模型，主要是为了让模型的权重和相关的 CUDA 内存分配都准备好，避免在正式推理时出现不确定的延迟或内存不足的问题
    def warmup_model(self):
        # 清空显存并重置峰值显存的记录
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # 使用最大负载来得到实际运行时的显存峰值占用
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)] # 由于这次推理只需要起到预热的效果就行了，所以可以任意赋值
        self.run(seqs, True) # 跑一遍refill 阶段
        torch.cuda.empty_cache() # 清空显存

    # 推理前为模型分配 KV Cache
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        # config.num_kvcache_blocks = (可用显存字节) // block_bytes
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0 # 断言必须 > 0，保证至少能分配一块。
        # 第一个维度 2：K 和 V 两个缓存；num_hidden_layers：每层 Transformer 各自有一份 KV cache；num_kvcache_blocks：每层能存多少个 block；其余维度对应注意力机制的 shape
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        # 遍历 self.model.modules()，把每一层的 k_cache 和 v_cache 指针指向这块大张量的切片，这样模型在推理时就能直接读写这块共享的 KV cache，而不需要单独为每层分配小块显存。
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens # 这里减掉已经缓存的 token 是因为只需要当前的 query 来计算 value，往前的 query 已经不需要了
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q) # 记录当前 seq 的终止位置，起始位置就是上一个 seq 的终止位置
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q) # 记录 query 最大长度
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            # block_table 为空则跳过，这是因为在 warmup_model 中还没有分配实际的缓存块表
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks): # 遍历未 cache 的物理块
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1: # 如果不是最后一个块
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end))) # 添加计算出的物理 slot 索引范围
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        # pin_memory=True 表示使用锁页内存，可以加速从 CPU 到 GPU 的数据传输
        # non_blocking=True 表示可以不用等待数据传输完成就可以继续执行后续代码
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq)) # 上下文总长度
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs) # 此时必然已经分配了 kv blocks 了，所以一定会需要对齐
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables) # 由于 decode 阶段固定只生成一个 token，因此 cu_seqlens_q, max_seqlen_q 等变量不再需要
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512: 
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None # 采样参数准备
        logits = self.run_model(input_ids, positions, is_prefill) # 模型前向传播
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None # Token 采样
        reset_context() # 上下文清理
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64) # 当前要推理的 token ID
        positions = torch.zeros(max_bs, dtype=torch.int64) # 每个 token 在序列中的位置
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32) # 指向 KV cache 中具体存储位置的映射
        context_lens = torch.zeros(max_bs, dtype=torch.int32) # 每个序列的上下文长度
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32) # 每个序列对应的 KV cache block 表
        outputs = torch.zeros(max_bs, hf_config.hidden_size) # 模型输出的隐藏状态
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16)) 
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs): # 从大到小遍历 batch size，这样最大的内存需求会首先被满足
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs]) # 设置上下文信息
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup，避免编译、初始化等一次性操作被记录，提高录制的 cuda graph 效率
            with torch.cuda.graph(graph, self.graph_pool): # 这个代码块中执行的所有CUDA操作都不会立即在GPU上运行，而是会被记录到 graph 对象中
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool() # 在成功捕获第一个 graph 后（即 bs 最大的 graph），保存其内存池，在后续继续捕获更小的 bs 的 graph 时共享，节约显存
            self.graphs[bs] = graph
            torch.cuda.synchronize() # 等待当前 graph 完全捕获
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
