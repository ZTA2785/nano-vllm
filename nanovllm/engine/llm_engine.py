import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)} # 拿到 Config 这个 dataclass 的所有字段定义
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields} # 拿到用户传入的 kwargs 中与 Config 字段同名的部分，作为实例化 Config 的参数
        config = Config(model, **config_kwargs) # 用 model（位置参数）和筛选后的 config_kwargs（关键字参数）实例化 Config
        self.ps = [] # 所有子进程对象，方便管理（终止、join 等）
        self.events = []  # 所有事件对象，用于进程间同步
        ctx = mp.get_context("spawn") # 得到多进程上下文，使用 spawn 模式来避免 fork 时的 CUDA 问题
        for i in range(1, config.tensor_parallel_size): # 根据 tensor_parallel_size 启动多个 ModelRunner 进程，i 从 1 开始，因为主进程也会运行一个 ModelRunner（tensor_parallel_size=1 时只有主进程运行 ModelRunner）
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event)) # 创建子进程，将会运行 ModelRunner(config, i, event)
            process.start()
            self.ps.append(process)
            self.events.append(event) # 收集子进程和事件对象
        
        # 然后初始化 model_runner、tokenizer、scheduler：
        self.model_runner = ModelRunner(config, 0, self.events) # 注意这里需要初始化当前进程的 model_runner，最后一个参数传入的是 self.events 列表而非单个事件，这是因为 rank 0 负责协调，能访问所有事件
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        
        # 最后使用 atexit 包将 self.exit 方法注册为一个清理函数，保证程序退出时资源正常释放：
        atexit.register(self.exit)

    # 清理函数。一是让 model_runner 退出并释放 model_runner，二是将所有子进程对象进行 join 方法保证结束
    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    # 添加一个请求到 scheduler 中，输入 prompt 和 sampling_params，prompt 可以是字符串也可以是已经编码成 token_id 的列表
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule() # scheduler 负责规划每一步需要向前推理的请求，同时返回一个是否处于 prefill 阶段的 bool 值
        token_ids = self.model_runner.call("run", seqs, is_prefill) # 让实际模型跑一步得到新的 token ids
        self.scheduler.postprocess(seqs, token_ids) # 后处理
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished] # 只返回已经完成的请求
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs) # 这里如果是 decode 阶段则会返回一个负数，和上面的 generate 方法中对应的 if-else 判断语句逻辑相符
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)  # 这里处理不是列表的情况，默认所有序列生成参数相同
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp) # 调用上一部分提到的 add_request 方法
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step() # step 方法，执行一步推理，返回输出和生成的 token 数量
            if use_tqdm:
                if num_tokens > 0: # 约定大于 0 时是 prefill，小于 0 时是 decode
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            # 下面开始处理返回值，由于这些值和 step 方法相关，建议先看 step 的实现
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
