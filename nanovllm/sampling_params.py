from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0 # 在 attention 操作的 softmax 中使用的温度，控制生成下一个 token 的混乱程度，值越低（越接近 0）则结果越确定，值越高（1 或更高）则回答更多样。
    max_tokens: int = 64 # 控制最长回答长度
    ignore_eos: bool = False # 是否忽略 eos（end of sequence），如果不忽略（值为 False）的话，当一个 token 的 id 是代表 eos 时，会停止当前请求的生成。

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
