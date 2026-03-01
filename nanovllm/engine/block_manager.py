from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id # 标识每个 block
        self.ref_count = 0 # 当前 block 被引用了多少次（例如 prefix cache）
        self.hash = -1 # 根据 token_ids 计算出的 hash 值，默认 -1 表示当前块尚未生成完成
        self.token_ids = [] # 当前 block 包含的 token ID 列表

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod # 计算每个块的 hash 值
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # 分配当前未被引用的块
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0 # 分配这个块的前提是这个块当前是未被引用的
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # 能否再分配一个 block
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table # 没有被分配过
        h = -1
        cache_miss = False # 是否 cache 命中
        for i in range(seq.num_blocks):
            token_ids = seq.block(i) # 获取第 i 个 block 的 token_ids
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1 # 如果 token_ids 可以构成一个完整的 block，则根据 token_ids 来为这个 block 计算一个 hash 值用于辨认，否则如果是不完整的块，则默认 hash 值为 -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss: # 缓存不命中则新分配一个 block
                block_id = self.free_block_ids[0] # 注意这里只“引用”了第一个空闲块的 id，没有真正取出来
                block = self._allocate_block(block_id) # 在这里正式分配
            else:  # 缓存命中
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids: # 如果这个命中的块正在被其他请求使用
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else: # 否则这个块是空闲的，需要通过 _allcate_block 来重新申明分配这个块
                    block = self._allocate_block(block_id)
            # 如果是一个完整块，则更新 hash 值和 token_ids
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            # 更新 seq 的 block table
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table): # 倒序释放
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    # 是否能够再追加一个 token
    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        # 刚好多出一个 token，则新分配一个 block
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        # 刚好满一个 block，则为该 block（seq 最后一个 block）更新 hash 值，并记录到 hash_to_block_id 表中
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        # 否则块处于中间态，hash 值必定是 -1
        else:
            assert last_block.hash == -1
