import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384 # 每次送入模型的 token 数量上限，必须大于等于 max_model_len
    max_num_seqs: int = 512 # 每次送入模型的序列数量上限，必须大于等于 max_num_batched_tokens // max_model_len
    max_model_len: int = 4096 # 模型最大上下文长度，必须小于等于 hf_config.max_position_embeddings
    gpu_memory_utilization: float = 0.9 # GPU 显存利用率，取值范围 (0, 1)，建议设置为 0.9 或更小，以避免 OOM
    tensor_parallel_size: int = 1 # 张量并行的大小，必须是 1、2、4、8 之一
    enforce_eager: bool = False # 是否强制使用 eager 模式（不使用共享内存和多进程），适用于调试和不支持多进程的环境，与之相对的是 graph mode
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256 # kvcache 的 block 大小，单位是 token，必须是 256 的倍数
    num_kvcache_blocks: int = -1 # kvcache 的 block 数量，-1 表示根据 max_model_len 自动计算

    def __post_init__(self): # 在 dataclass 修饰器之后运行，可以用来参数校验以及动态调整成员值等
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings) # 将用户设置的 max_model_len 与模型原生支持的最大上下文长度 (max_position_embeddings) 取最小值
        assert self.max_num_batched_tokens >= self.max_model_len # 确保“单次推理的总Token容量”至少能装下一条“最长的序列”。
