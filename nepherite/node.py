import asyncio
import os
import random
import time
from collections import defaultdict
from typing import Literal

from ipv8.community import CommunitySettings
from ipv8.messaging.payload_dataclass import dataclass
from ipv8.types import Peer

# from rocksdict import Options, Rdict
from nepherite.base import Blockchain, message_wrapper
from nepherite.merkle import MerkleTree
from nepherite.puzzle import HashNoncePuzzle as Puzzle
from nepherite.utils import logging, sha256

BLOCK_SIZE = 4
BLOCK_REWARD = 100
BLOCK_DIFFICULTY = 6
RETRY_COUNT = 3


@dataclass
class Utxo:
    address: bytes
    amount: int


@dataclass
class TransactionPayload:
    timestamp: int
    output: list[Utxo]


@dataclass(msg_id=1)
class Transaction:
    payload: TransactionPayload
    pk: bytes
    sign: bytes


@dataclass(msg_id=2)
class BlockHeader:
    seq_num: int
    prev_block_hash: bytes
    merkle_root_hash: bytes
    timestamp: int
    difficulty: int
    nonce: int


@dataclass(msg_id=3)
class Block:
    header: BlockHeader
    transactions: list[Transaction]


@dataclass(msg_id=4)
class PullBlockRequest:
    block_hash: bytes
    address: tuple[str, int]
    nonce: int


@dataclass(msg_id=5)
class PullBlockAck:
    nonce: int


class NepheriteNode(Blockchain):
    def __init__(self, settings: CommunitySettings) -> None:
        super().__init__(settings)

        self.next_blocks: dict[bytes, list[bytes]] = defaultdict(list)
        self.blocks: dict[bytes, Block] = set()
        self.mempool: dict[bytes, Transaction] = {}
        self.blocks_info: dict[bytes, BlockHeader] = {}
        self.last_seq_num = 0
        self.events: dict[int, asyncio.Event] = {}

        # options = Options(raw_mode=False)
        # self.chainstate = Rdict("data/chainstate.db", options=options)
        self.chainstate: dict[bytes, int] = defaultdict(0)
        self.is_mining = False

        self.add_message_handler(BlockHeader, self.on_block_header)
        self.add_message_handler(PullBlockRequest, self.on_pull_block_request)
        self.add_message_handler(Transaction, self.on_transaction)
        self.add_message_handler(Block, self.on_block)

    def on_start(self):
        self.register_anonymous_task(
            "create_dummy_transaction", self.create_dummy_transaction, interval=5
        )
        self.register_anonymous_task(
            "start_to_create_block", self.start_to_create_block, interval=3
        )

    def __log(self, level: Literal["info", "warn", "error", "debug"], msg: str):
        match level:
            case "info":
                logging.info(f"Node {self.my_peer.mid.hex()[:6]}: {msg}")
            case "warn":
                logging.warn(f"Node {self.my_peer.mid.hex()[:6]}: {msg}")
            case "error":
                logging.error(f"Node {self.my_peer.mid.hex()[:6]}: {msg}")
            case "debug":
                logging.debug(f"Node {self.my_peer.mid.hex()[:6]}: {msg}")

    def start_to_create_block(self):
        self.__log("info", "Start to create block")

        if self.is_mining:
            self.__log("warn", "Already mining")
            return

        if len(self.mempool) >= BLOCK_SIZE:
            self.__log("info", "Start mining")
            block = self.mine_block()
            self.__log("info", "Block mined")

            for peer in self.get_peers():
                self.ez_send(peer, block)
                self.__log("info", f"Block sent to {peer.mid.hex()[:6]}")

    def create_dummy_transaction(self):
        peer = random.choice(self.get_peers())
        out = [Utxo(peer.mid, 100)]

        for peer in self.get_peers():
            if peer.mid != self.my_peer.mid:
                tx = self.make_and_sign_transaction(out)
                self.ez_send(peer, tx)
                self.__log("debug", f"Sent tx to {peer.mid.hex()[:6]}")

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

    def make_and_sign_transaction(self, output: list[Utxo]) -> Transaction:
        payload = TransactionPayload(
            timestamp=int(time.monotonic_ns() // 1_000),
            output=output,
        )
        blob = self.serializer.pack_serializable(payload)
        sign = self.crypto.create_signature(self.my_peer.key, blob)
        pk = self.my_peer.key.pub().key_to_bin()
        return Transaction(payload, pk, sign)

    def get_public_key_by_mid(self, mid):
        for _, p in self.nodes.items():
            if p.mid == mid:
                return p.public_key

    def verify_transaction(
        self, transaction: Transaction, coinbase: bool = False
    ) -> bool:
        pk = self.crypto.key_from_public_bin(transaction.pk)
        addr = pk.key_to_hash()
        blob = self.serializer.pack_serializable(transaction.payload)
        if not self.crypto.is_valid_signature(pk, blob, transaction.sign):
            return False
        sum = 0  # noqa: A001
        for utxo in transaction.payload.output:
            sum += utxo.amount  # noqa: A001
        if coinbase:
            return sum == BLOCK_REWARD
        amount = self.chainstate[addr]
        return amount >= sum

    def verify_block(self, block: Block) -> bool:
        header = block.header
        blob = self.serializer.pack_serializable(header)

        # Puzzle solved?
        if not Puzzle.verify(blob, header.nonce):
            return False
        # No double spending?
        if any(not self.verify_transaction(tx) for tx in block.transactions):
            return False
        # Correct difficulty?
        if block.header.difficulty != BLOCK_DIFFICULTY:
            return False
        # Correct previous block?
        if prev_block := self.blocks_info.get(header.prev_block_hash):
            if prev_block.seq_num + 1 != header.seq_num:
                return False
        else:
            return False
        # Correct merkle root?
        tree = MerkleTree(block.transactions)
        if tree.root.hash != header.merkle_root_hash:
            return False
        return True

    def check_if_tx_in_mempool(self, tx: Transaction):
        return self.mempool.get(tx.sign) is not None

    @message_wrapper(Transaction)
    def on_transaction(self, peer: Peer, transaction: Transaction) -> None:
        # TODO: remove this debug
        peer_id = self.node_id_from_peer(peer)
        self.__log("info", f"Transaction from {peer_id} received")

        if self.check_if_tx_in_mempool(transaction):
            self.__log("info", f"Transaction from {peer_id} is already in mempool")
            return
        if not self.verify_transaction(transaction):
            self.__log("warn", f"Transaction from {peer_id} is invalid")
            return

        self.__log("info", f"Transaction from {peer_id} is valid")
        self.mempool.append(transaction)
        for u in self.get_peers():
            if u.mid != peer.mid:
                self.ez_send(u, transaction)

    @message_wrapper(BlockHeader)
    def on_block_header(self, peer: Peer, block_header: BlockHeader) -> None:
        block_hash = self.get_block_hash(block_header)
        if block_hash in self.blocks_info:
            return

        # Broadcast block header over the network
        self.next_blocks[block_header.prev_block_hash].append(block_header)
        self.blocks_info[block_hash] = block_header
        for u in self.get_peers():
            self.ez_send(u, block_header)

        # If we that block seams to be the next one, request the full block
        if block_header.seq_num >= self.last_seq_num:
            nonce = int.from_bytes(os.urandom(8))
            self.ez_send(peer, PullBlockRequest(block_hash, self.my_peer.mid, nonce))

            # partially rollback
            # find the last block that is in the chain
            # TODO: Implement rollback

    def get_transaction_trace(self, transaction: Transaction) -> dict[bytes, int]:
        ret = {}
        out = 0
        for utxo in transaction.payload.output:
            addr = utxo.address
            amount = utxo.amount
            ret[addr] += amount
            out += amount
        pk = self.crypto.key_from_public_bin(transaction.pk)
        addr = pk.key_to_hash()
        ret[addr] -= out
        return ret

    @message_wrapper(Block)
    def on_block(self, peer: Peer, block: Block) -> None:
        self.__log("info", f"Block {block.header.seq_num} received")
        # if not self.verify_block(block):
        #     return

        # self.__log("info", f"Block {block.header.seq_num} verified")
        # # block_hash = self.get_block_hash(block.header)  # noqa: A001
        # # self.blocks[block_hash] = block
        # # self.blocks_info[block_hash] = block.header
        # # self.save_block(block)
        # # self.last_seq_num = block.header.seq_num

        # TODO: Update chainstate

    def build_coinbase_transaction(self) -> Transaction:
        output = [Utxo(self.my_peer.mid, BLOCK_REWARD)]
        return self.make_and_sign_transaction(output)

    def build_block(self) -> Block:
        if self.last_seq_num != 0:
            previous_block = self.load_block(self.last_seq_num)
            prev_block_hash = previous_block.header.prev_block_hash
        else:
            # Genesis block
            prev_block_hash = b"\x00" * 32

        transactions = [self.build_coinbase_transaction()]

        # Include transactions from mempool
        for tx in self.mempool:
            if len(transactions) >= BLOCK_SIZE:
                break
            ret = self.get_transaction_trace(tx)
            if all(self.chainstate[k] + v >= 0 for k, v in ret.items()):
                transactions.append(tx)
            self.mempool.pop(tx.sign)

        tree = MerkleTree(transactions)
        header = BlockHeader(
            seq_num=self.last_seq_num + 1,
            prev_block_hash=prev_block_hash,
            merkle_root_hash=tree.root.hash,
            timestamp=time.monotonic_ns() // 1_000,
            difficulty=BLOCK_DIFFICULTY,
            nonce=0,
        )
        return Block(header, transactions)

    def mine_block(self) -> Block:
        block = self.build_block()
        blob = self.serializer.pack_serializable(block.header)
        answer = Puzzle.compute(blob)
        block.header.nonce = answer
        return block

    @message_wrapper(PullBlockAck)
    def on_pull_block_response(self, peer: Peer, response: PullBlockAck) -> None:
        if event := self.events.get(response.nonce):
            event.set()

    @message_wrapper(PullBlockRequest)
    async def on_pull_block_request(
        self, peer: Peer, request: PullBlockRequest
    ) -> None:
        block_hash = request.block_hash
        if block_hash not in self.blocks:
            for u in self.get_peers():
                self.ez_send(u, request)
            return

        block = self.blocks[block_hash]
        self.events[request.nonce] = asyncio.Event()
        cnt = RETRY_COUNT

        while cnt > 0:
            self.walk_to(request.address)
            async with asyncio.timeout(5):
                await self.events[request.nonce].wait()
                self.ez_send(peer, block)
            if self.events[request.nonce].is_set():
                break
            cnt -= 1
