import time
from collections import defaultdict
from dataclasses import dataclass

from ipv8.community import CommunitySettings
from ipv8.messaging.payload_dataclass import dataclass as payload_dataclass
from ipv8.types import Peer
from rocksdict import Options, Rdict

from nepherite.base import Blockchain, message_wrapper
from nepherite.merkle import MerkleTree
from nepherite.puzzle import Puzzle
from nepherite.utils import sha256

BLOCK_SIZE = 16
BLOCK_REWARD = 100
BLOCK_DIFFICULTY = 6


@dataclass
class Utxo:
    address: str
    amount: int


@payload_dataclass(msg_id=1)
class Transaction:
    timestamp: int
    input: list[Utxo]
    output: list[Utxo]


@payload_dataclass(msg_id=2)
class BlockHeader:
    seq_num: int
    version: int
    prev_block_hash: bytes
    merkle_root_hash: bytes
    timestamp: int
    difficulty: int
    nonce: int


@payload_dataclass(msg_id=3)
class Block:
    header: BlockHeader
    transactions: list[Transaction]


@payload_dataclass(msg_id=4)
class PullBlockRequest:
    block_hash: bytes
    peer_id: bytes


class NepheriteNode(Blockchain):
    def __init__(self, settings: CommunitySettings) -> None:
        super().__init__(settings)

        self.next_blocks: dict[bytes, list[bytes]] = defaultdict(list)
        self.mempool: list[Transaction] = []
        self.blocks: dict[bytes, BlockHeader] = {}
        self.last_seq_num = 0

        options = Options(raw_mode=False)
        self.chainstate = Rdict("data/chainstate.db", options=options)

        self.add_message_handler(BlockHeader, self.on_block_header)
        self.add_message_handler(PullBlockRequest, self.on_pull_block_request)
        self.add_message_handler(Transaction, self.on_transaction)

    def get_block_hash(self, header: BlockHeader) -> bytes:
        blob = self.serializer.pack_serializable(header)
        return sha256(blob)

    def save_block(self, block: Block) -> bool:
        blob = self.serializer.pack_serializable(block)
        with open(f"data/blocks/{block.seq_num}", "wb") as f:
            f.write(blob)
        return True

    def load_block(self, seq_num: int) -> Block:
        with open(f"data/blocks/{seq_num}", "rb") as f:
            blob = f.read()
        return self.serializer.unpack_serializable(blob, Block)

    def verify_transaction(self, transaction: Transaction) -> bool:
        sum_in = 0
        sum_out = 0
        for utxo in transaction.input:
            sum_in += utxo.amount
            if utxo.address not in self.chainstate:
                return False
        for utxo in transaction.output:
            sum_out += utxo.amount

        # TODO: fee stuff (ask teacher)
        return sum_in >= sum_out
    
    def verify_block(self, block: Block) -> bool:
        header = block.header
        
        block_hash = self.get_block_hash(header)


    def on_transaction(self, peer: Peer, transaction: Transaction) -> None:
        if not self.verify_transaction(transaction):
            return

        self.mempool.append(transaction)
        # broadcast transaction
        for u in self.get_peers():
            self.ez_send(u, transaction)

    @message_wrapper(BlockHeader)
    def on_block_header(self, peer: Peer, block_header: BlockHeader) -> None:
        block_hash = self.get_block_hash(block_header)
        if block_hash in self.blocks:
            return

        self.next_blocks[block_header.prev_block_hash].append(block_header)
        self.blocks[block_hash] = block_header
        for u in self.get_peers():
            self.ez_send(u, block_header)

    def build_block(self) -> Block:
        previous_block = self.load_block(self.last_seq_num)
        transactions = self.mempool[:BLOCK_SIZE]
        tree = MerkleTree(transactions)

        header = BlockHeader(
            seq_num=self.last_seq_num + 1,
            version=1,
            prev_block_hash=previous_block.header.prev_block_hash,
            merkle_root_hash=tree.root.hash,
            timestamp=time.monotonic_ns() // 1_000,
            difficulty=BLOCK_DIFFICULTY,
            nonce=0,
        )

        return Block(header, transactions)

    def mine_block(self) -> Block:
        block = self.build_block()
        blob = self.serializer.pack_serializable(block)
        answer = Puzzle.compute(blob)

        # TODO: Implement mining
        ...

    # @message_wrapper(PullBlockRequest)
    # def on_pull_block_request(self, peer: Peer, request: PullBlockRequest) -> None:
    #     block_hash = request.block_hash
    #     if block_hash not in self.blocks:
    #         self.parent[block_hash] = peer
    #         # broadcast pull request
    #         for u in self.get_peers():
    #             self.ez_send(u, request)
    #     block = self.blocks[block_hash]
    #     self.ez_send(peer, block)
