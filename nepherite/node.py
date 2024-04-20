import asyncio
import os
import random
import time
from collections import defaultdict
from threading import Lock

from ipv8.community import CommunitySettings
from ipv8.messaging.payload_dataclass import dataclass
from ipv8.types import Peer

# from rocksdict import Options, Rdict
from nepherite.base import Blockchain, message_wrapper
from nepherite.config import (
    BLOCK_REWARD,
    BLOCK_SIZE,
    BLOCK_TTL,
    CHAIN_GAP_SIZE,
)
from nepherite.merkle import MerkleTree
from nepherite.puzzle import DIFFICULTY as BLOCK_DIFFICULTY
from nepherite.puzzle import HashNoncePuzzle as Puzzle
from nepherite.utils import sha256


@dataclass
class TxOut:
    address: bytes
    amount: int


@dataclass
class TransactionPayload:
    nonce: int
    output: list[TxOut]


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


class NepheriteNode(Blockchain):
    def __init__(self, settings: CommunitySettings) -> None:
        super().__init__(settings)

        # self.blocks: dict[int, list[Block]] = defaultdict(list)
        self.mempool: dict[bytes, Transaction] = {}
        self.blockset: dict[bytes, Block] = {}
        self.invalid_blocks: set[bytes] = set()
        self.current_seq_num = 0
        self.current_block_hash = b"\x00" * 32
        self.ttl: dict[bytes, int] = defaultdict(lambda: BLOCK_TTL)

        # options = Options(raw_mode=False)
        # self.chainstate = Rdict("data/chainstate.db", options=options)
        self.chainstate: dict[bytes, int] = defaultdict(int)
        self.lock_mining = Lock()
        self.mining_cancellation = asyncio.Event()

        self.seed_genesis_block()

        self.add_message_handler(Transaction, self.on_transaction)
        self.add_message_handler(Block, self.on_block)
        self.add_message_handler(PullBlockRequest, self.on_pull_block_request)

    def seed_genesis_block(self):
        genesis_block = Block(
            BlockHeader(
                seq_num=0,
                prev_block_hash=b"\x00" * 32,
                merkle_root_hash=b"\x00" * 32,
                timestamp=0,
                difficulty=BLOCK_DIFFICULTY,
                nonce=0,
            ),
            [],
        )
        self.save_block(genesis_block)
        self.current_block_hash = self.get_block_hash(genesis_block.header)
        self.genesis_block_hash = self.current_block_hash
        self.current_seq_num = 0
        self.blockset[self.current_block_hash] = genesis_block

    def on_start(self):
        self.register_anonymous_task(
            "create_dummy_transaction", self.create_dummy_transaction, interval=5
        )
        self.register_anonymous_task("mining_monitor", self.mining_monitor)

    async def mining_monitor(self):
        self._log("info", "Mining monitor started")
        while True:
            mining = self.register_executor_task(
                "start_to_create_block", self.start_to_create_block
            )
            cancellation = self.mining_cancellation.wait()
            _, pending = await asyncio.wait(
                [mining, asyncio.ensure_future(cancellation)],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for pending_task in pending:
                pending_task.cancel()

            self._log("info", "Mining finished, waiting for next round")
            self.mining_cancellation.clear()
            await asyncio.sleep(10.0)

    def start_to_create_block(self):
        self._log("info", "Start mining")
        try:
            last_seq_num = self.current_seq_num
            block = self.mine_block()

            # TODO: remove this sh** (is ugly but works for now)
            if last_seq_num != self.current_seq_num:
                self._log("warn", "Block already mined")
                return            
            with self.lock_mining:
                self.current_seq_num = max(self.current_seq_num, block.header.seq_num)
                self.current_block_hash = self.get_block_hash(block.header)
                self.blockset[self.current_block_hash] = block
                self._log("info", f"Block {block.header.seq_num} mined")
            for peer in self.get_peers():
                self.ez_send(peer, block)
                self._log("info", f"Block sent to {peer.mid.hex()[:6]}")
        except Exception as e:
            self._log("error", str(e))

    def create_dummy_transaction(self):
        cnt = self.chainstate[self.my_peer.mid]
        if cnt < 10:
            return

        peer = random.choice(self.get_peers())
        out = [TxOut(peer.mid, min(cnt, 10))]

        for peer in self.get_peers():
            if peer.mid != self.my_peer.mid:
                tx = self.make_and_sign_transaction(out)
                self.ez_send(peer, tx)
                self._log("debug", f"Sent tx to {peer.mid.hex()[:6]}")

    def get_block_hash(self, header: BlockHeader) -> bytes:
        blob = self.serializer.pack_serializable(header)
        return sha256(blob)

    def save_block(self, block: Block) -> bool:
        blob = self.serializer.pack_serializable(block)
        with open(f"data/blocks/{block.header.seq_num}", "wb") as f:
            f.write(blob)
        return True

    def load_block(self, seq_num: int) -> Block:
        with open(f"data/blocks/{seq_num}", "rb") as f:
            blob = f.read()
        return self.serializer.unpack_serializable(blob, Block)

    def make_and_sign_transaction(self, output: list[TxOut]) -> Transaction:
        payload = TransactionPayload(
            nonce=int.from_bytes(os.urandom(4)),
            output=output,
        )
        blob = self.serializer.pack_serializable(payload)
        sign = self.crypto.create_signature(self.my_peer.key, blob)
        pk = self.my_peer.key.pub().key_to_bin()
        return Transaction(payload, pk, sign)

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

    def stateless_block_verification(self, block: Block) -> bool:
        header = block.header

        # TODO: remove this sh** (is ugly but works for now)
        nonce = block.header.nonce
        block.header.nonce = 0

        blob = self.serializer.pack_serializable(header)

        # 1. Check the proof of work (nonce + previous hash = answer)
        if not Puzzle.verify(blob, nonce):
            return False

        block.header.nonce = nonce

        # 2. Check the merkle root hash
        tree = MerkleTree([tx.sign for tx in block.transactions])
        if tree.root.hash != header.merkle_root_hash:
            return False

        return True

    @message_wrapper(Transaction)
    def on_transaction(self, peer: Peer, transaction: Transaction) -> None:
        # TODO: remove this debug
        peer_id = peer.mid.hex()[:6]
        self._log("info", f"Transaction from {peer_id} received")

        if self.mempool.get(transaction.sign) is not None:
            self._log("info", f"Transaction from {peer_id} is already in mempool")
            return
        if not self.verify_transaction(transaction):
            self._log("warn", f"Transaction from {peer_id} is invalid")
            return

        self._log("info", f"Transaction from {peer_id} is valid")
        self.mempool[transaction.sign] = transaction
        for u in self.get_peers():
            if u.mid != peer.mid:
                self.ez_send(u, transaction)

    @message_wrapper(PullBlockRequest)
    def on_pull_block_request(self, peer: Peer, request: PullBlockRequest) -> None:
        self._log("info", f"Pull Request received from {peer.mid.hex()[:6]}")
        block_hash = request.block_hash
        if block_hash not in self.blockset:
            self._log("warn", "Block is not in the my chain")
            for near in self.get_peers():
                if near.mid == peer.mid:
                    continue
                self.ez_send(near, PullBlockRequest(block_hash))
            return
        self._log("info", "Block sent")
        self.ez_send(peer, self.blockset[block_hash])

    def get_transaction_deltas(self, transaction: Transaction) -> dict[bytes, int]:
        deltas = defaultdict(int)
        out = 0
        for utxo in transaction.payload.output:
            addr = utxo.address
            amount = utxo.amount
            deltas[addr] += amount
            out += amount
        pk = self.crypto.key_from_public_bin(transaction.pk)
        addr = pk.key_to_hash()
        deltas[addr] -= out
        return deltas

    def rollback(self, v: bytes) -> bytes:
        u = self.current_block_hash
        path = []
        while self.blockset[u].header.seq_num < self.blockset[v].header.seq_num:
            path.append(v)
            v = self.blockset[v].header.prev_block_hash
            if v not in self.blockset:
                return None, {}, path
        deltas = defaultdict(int)
        while u != v:
            for tx in self.blockset[u].transactions:
                dt = self.get_transaction_deltas(tx)
                self.apply_transaction(dt, deltas, -1)

            path.append(v)
            u = self.blockset[u].header.prev_block_hash
            v = self.blockset[v].header.prev_block_hash
            if u not in self.blockset or v not in self.blockset:
                return None, {}, path
        return u, deltas, path

    @message_wrapper(Block)
    def on_block(self, peer: Peer, block: Block) -> None:
        self._log(
            "info", f"Block {block.header.seq_num} received from {peer.mid.hex()[:6]}"
        )

        if not self.stateless_block_verification(block):
            self._log("warn", f"Block {block.header.seq_num} is invalid")
            return

        block_hash = self.get_block_hash(block.header)
        self.blockset[block_hash] = block
        self.ttl[block_hash] = max(self.ttl[block_hash] - 1, 0)

        if self.ttl[block_hash] > 0:
            # Broadcast block to other peers
            for near in self.get_peers():
                if near.mid == peer.mid:
                    continue
                self.ez_send(near, block)
                self._log("info", f"Block sent to {near.mid.hex()[:6]}")

        if block.header.prev_block_hash not in self.blockset:
            self._log("warn", "Block is not in the chain")
            self.ez_send(peer, PullBlockRequest(block.header.prev_block_hash))
            return
        else:
            if block.header.seq_num >= self.current_seq_num + CHAIN_GAP_SIZE:
                with self.lock_mining:
                    # TODO: implement chain reorganization
                    self._log("info", "Chain reorganization")
                    lca, deltas, path = self.rollback(block_hash)

                    if lca is None:
                        self._log("warn", "Failed to find LCA")
                        self.ez_send(peer, PullBlockRequest(path[-1]))
                        self._log("warn", "Requested last parent to solve LCA")
                        return

                    valid_chain = True
                    for block_path in path:
                        if valid_chain:
                            for tx in self.blockset[block_path].transactions:
                                dt = self.get_transaction_deltas(tx)
                                if all(
                                    self.chainstate[addr] + value + deltas[addr] >= 0
                                    for addr, value in dt.items()
                                ):
                                    self.apply_transaction(dt, deltas)
                                else:
                                    self._log("warn", "Invalid transaction")
                                    # TODO: mark block as invalid
                                    self.invalid_blocks.add(block_path)
                                    self.blockset.pop(block_path)
                                    valid_chain = False
                                    break
                        else:
                            self.invalid_blocks.add(block_path)
                            self.blockset.pop(block_path)
                    if valid_chain:
                        self._log("warn", "Cancelling mining")
                        self.mining_cancellation.set()

                        # Rollback transactions from mempool
                        ptr = self.current_block_hash
                        while lca != ptr:
                            for tx in self.blockset[ptr].transactions:
                                if tx.sign in self.mempool:
                                    self.mempool.pop(tx.sign)
                            ptr = self.blockset[ptr].header.prev_block_hash

                        # Fast forward transactions from mempool
                        ptr = block_hash
                        while ptr != lca:
                            for tx in self.blockset[ptr].transactions:
                                self.mempool[tx.sign] = tx
                            ptr = self.blockset[ptr].header.prev_block_hash

                        self.current_block_hash = block_hash
                        self.current_seq_num = block.header.seq_num

                        # Update chainstate (UTXOs)
                        self.apply_transaction(deltas, self.chainstate)
                        self._log("info", "Chain reorganization completed")

    def build_coinbase_transaction(self) -> Transaction:
        output = [TxOut(self.my_peer.mid, BLOCK_REWARD)]
        return self.make_and_sign_transaction(output)

    def apply_transaction(
        self, deltas: dict[bytes, int], chain_state: dict[bytes, int], sign: int = 1
    ):
        for address, value in deltas.items():
            chain_state[address] += value * sign

    def build_valid_transactions_for_block(self) -> list[Transaction]:
        """
        Get valid transactions from mempool
        Delete invalid transactions from mempool while iterating
        Returns:
            list[Transaction]: List of valid transactions for the next block
        """
        valid_transactions = []
        deleted_transactions = []
        for tx in self.mempool.values():
            if len(valid_transactions) >= BLOCK_SIZE - 1:
                break
            deltas = self.get_transaction_deltas(tx)
            if all(
                self.chainstate[address] + value >= 0
                for address, value in deltas.items()
            ):
                valid_transactions.append(tx)
                self.apply_transaction(deltas, self.chainstate)
            deleted_transactions.append(tx)
        for tx in deleted_transactions:
            self.mempool.pop(tx.sign)
        return valid_transactions

    def build_block(self) -> Block:
        transactions = [self.build_coinbase_transaction()]
        # Include transactions from mempool
        with self.lock_mining:
            transactions += self.build_valid_transactions_for_block()
        tree = MerkleTree([tx.sign for tx in transactions])

        header = BlockHeader(
            seq_num=self.current_seq_num + 1,
            prev_block_hash=self.current_block_hash,
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
