from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque() # 等待被执行的序列
        self.running: deque[Sequence] = deque() # 正在推理的序列

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0] # 得到 waiting 队列中第一个，但暂时没有移出 waiting 队列
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq): # 如果超出最大限度，直接停止 schedule
                break
            num_seqs += 1
            self.block_manager.allocate(seq) # 分配新的 kv cache 块
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft() # 此时才将该请求移出 waiting 队列
            self.running.append(seq)
            scheduled_seqs.append(seq)
        # 由于上面进入 scheduled_seqs 的请求只能是 waiting 序列中的前若干项，所以必定是 prefill 阶段，返回的第二个参数为 True
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft() # 取出 running 队列的最左边一个请求
            # 如果不能添加这个请求，则：
            while not self.block_manager.can_append(seq):
                # 如果已经有在 decode 阶段的请求，则抢占 running 队列中的最右边一个
                if self.running:
                    self.preempt(self.running.pop())
                # 否则“抢占”自己，即将 seq 放回 waiting 队列的最左边
                else:
                    self.preempt(seq)
                    break
            # 如果能够直接将 seq 添加进 running 队列
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    # 抢占资源
    def preempt(self, seq: Sequence): 
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    # 用于 step 后后处理请求和新生成的 tokens。
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id) # 添加新 token
            # 如果当前 token 是 eos（end of sequence）且不忽略 eos，或者已经生成的 token 数达到了 max_tokens 的限制，则将该序列标记为 FINISHED，并释放其占用的资源。
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens: 
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
