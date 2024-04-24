import asyncio
import os
import random
import time
from collections import defaultdict
from threading import Lock
from typing import Tuple, Dict, Any, List

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
from enum import Enum

Token = {
    'NEPHERITE' : 1,
    'DEX_TOKEN' : 2,
    'SHIBA_INU' : 3,
}
@dataclass
class TxOut:
    address: bytes
    amount: int

@dataclass
class TransactionPayload:
    nonce: int
    output: list[TxOut]
    token: int


@dataclass
class AddLiquidityPayload:
    nonce: int
    value: int

@dataclass
class RemoveLiquidityPayload:
    nonce: int
    value: int

@dataclass
class ExchangePayload:
    nonce: int
    value: int
    token: bytes

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

# [A block, Messages to undo, deltas to apply]
LocalBlockOperation = tuple[Block, list[Transaction], dict[tuple[bytes, int], int]]


class NepheriteNode(Blockchain):
    def __init__(self, settings: CommunitySettings) -> None:
        super().__init__(settings)

        # Setup folders and stuff
        self.genesis_block_hash = None
        self.setup()

        # self.blocks: dict[int, list[Block]] = defaultdict(list)
        self.mempool: dict[bytes, Transaction] = {}
        self.blockset: dict[bytes, Block] = {}
        self.invalid_blocks: set[bytes] = set()
        self.current_seq_num = 0
        self.current_block_hash = b"\x00" * 32
        self.ttl: dict[bytes, int] = defaultdict(lambda: BLOCK_TTL)

        # options = Options(raw_mode=False)
        # self.chainstate = Rdict("data/chainstate.db", options=options)
        self.chainstate: defaultdict[tuple[bytes, int], int] = defaultdict(int)

        self.lock_mining = Lock()
        self.mining_cancellation = asyncio.Event()

        self.seed_genesis_block()

        self.add_message_handler(Transaction, self.on_transaction)
        self.add_message_handler(Block, self.on_block)
        self.add_message_handler(PullBlockRequest, self.on_pull_block_request)

        self._last_seq_num_for_tx = -1

    def setup(self):
        # Ensure the data directory exists
        os.makedirs("data/keys", exist_ok=True)
        os.makedirs("data/blocks", exist_ok=True)

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
        self.register_anonymous_task("report", self.report, interval=10)

    def report(self):
        self._log("info", f"Current block: {self.current_seq_num}")
        self._log("info", f"Current hash: {self.current_block_hash.hex()[:6]}")
        self._log(
            "info",
            f"Current chainstate: {[f'{k[0].hex()[:6]}:{k[1]} {v}' for k, v in self.chainstate.items()]}",
        )

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
            operation = self.build_and_mine_block()

            # TODO: remove this sh** (is ugly but works for now)
            if last_seq_num != self.current_seq_num:
                self._log("warn", "Block already mined")
                return

            # Commit block changes
            with self.lock_mining:
                block, tx_to_remove, deltas = operation
                self.current_seq_num = max(self.current_seq_num, block.header.seq_num)
                self.current_block_hash = self.get_block_hash(block.header)
                self.blockset[self.current_block_hash] = block

                # Apply deltas to chainstate
                self.apply_transaction(deltas, self.chainstate)
                # Remove transactions from mempool
                for tx in tx_to_remove:
                    self.mempool.pop(tx.sign)

                self._log("info", f"Block {block.header.seq_num} mined")
            for peer in self.get_peers():
                self.ez_send(peer, block)
                self._log("info", f"Block sent to {peer.mid.hex()[:6]}")
        except Exception as e:
            self._log("error", str(e))

    def create_dummy_transaction(self):
        if self._last_seq_num_for_tx == self.current_seq_num:
            return
        self._last_seq_num_for_tx = self.current_seq_num
        cnt = self.chainstate[self.my_peer.mid,Token['NEPHERITE']]
        if cnt < 10:
            return

        peer = random.choice(self.get_peers())  # nosec B311
        out = [TxOut(peer.mid, min(cnt, 10))]
        tx = self.make_and_sign_transaction(out)
        self.mempool[tx.sign] = tx  # add to mempool

        for peer in self.get_peers():
            if peer.mid != self.my_peer.mid:
                self.ez_send(peer, tx)
                self._log("debug", f"Sent tx with {cnt} coins to {peer.mid.hex()[:6]}")

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

    def make_and_sign_transaction(self, output: list[TxOut], token:int = 1) -> Transaction:
        payload = TransactionPayload(
            nonce=int.from_bytes(os.urandom(4)),
            output=output,
            token=token
        )
        blob = self.serializer.pack_serializable(payload)
        sign = self.crypto.create_signature(self.my_peer.key, blob)
        pk = self.my_peer.key.pub().key_to_bin()
        return Transaction(payload, pk, sign)

    def verify_sign_transaction(
        self, message: Transaction, coinbase: bool = False
    ) -> bool:
        pk = self.crypto.key_from_public_bin(message.pk)
        blob = self.serializer.pack_serializable(message.payload)
        if not self.crypto.is_valid_signature(pk, blob, message.sign):
            return False
        return True

    def stateless_block_verification(self, block: Block) -> bool:
        header = block.header

        # TODO: remove this sh** (is ugly but works for now)
        # 1. Check the proof of work (nonce + previous hash = answer)
        nonce = block.header.nonce
        block.header.nonce = 0  # remove nonce
        if not Puzzle.verify(self.get_block_hash(block.header), nonce):
            return False
        block.header.nonce = nonce  # rollback

        # 2. Check the merkle root hash
        tree = MerkleTree([tx.sign for tx in block.transactions])
        if tree.root.hash != header.merkle_root_hash:
            return False

        return True

    def on_add_liquidity(self, peer: Peer, message: Transaction) -> None:
        pass

    def on_remove_liquidity(self, peer: Peer, message: Transaction) -> None:
        pass

    def on_exchange(self, peer: Peer, message: Transaction) -> None:
        pass

    def undo_add_liquidity(self, message: Transaction) -> None:
        pass

    def undo_remove_liquidity(self, message: Transaction) -> None:
        pass

    def undo_exchange(self, message: Transaction) -> None:
        pass

    @message_wrapper(Transaction)
    def on_transaction(self, peer: Peer, transaction: Transaction) -> None:
        # TODO: remove this debug
        peer_id = peer.mid.hex()[:6]
        self._log("info", f"Transaction from {peer_id} received")

        if self.mempool.get(transaction.sign) is not None:
            self._log("info", f"Transaction from {peer_id} is already in mempool")
            return
        if not self.verify_sign_transaction(transaction):
            self._log("warn", f"Transaction from {peer_id} has an invalid signature")
            return

        self._log("info", f"Transaction from {peer_id} is valid signed")
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
        self.ttl[block_hash] = min(self.ttl[block_hash] + 1, 2)
        self.ez_send(peer, self.blockset[block_hash])

    def get_transaction_deltas(
        self, transaction: Transaction, coinbase: bool = False
    ) -> dict[tuple[bytes, int], int]:
        deltas = defaultdict(int)
        out = 0
        token = transaction.payload.token
        for utxo in transaction.payload.output:
            addr = utxo.address
            amount = utxo.amount
            deltas[addr,token] += amount
            out += amount
        if not coinbase:
            pk = self.crypto.key_from_public_bin(transaction.pk)
            addr = pk.key_to_hash()
            deltas[addr,token] -= out
        return deltas

    def rollback(self, v: bytes) -> tuple[None, dict[Any, Any], list[bytes]] | tuple[
        bytes, defaultdict[Any, int], list[bytes]]:
        u = self.current_block_hash
        path = []
        while self.blockset[u].header.seq_num < self.blockset[v].header.seq_num:
            path.append(v)
            v = self.blockset[v].header.prev_block_hash
            if v not in self.blockset:
                return None, {}, path
        deltas = defaultdict(int)
        while u != v:
            # undo coinbase transaction
            self.apply_transaction(
                self.get_transaction_deltas(
                    self.blockset[u].transactions[0], coinbase=True
                ),
                deltas,
                -1,
            )
            self.apply_transaction(
                self.get_transaction_deltas(
                    self.blockset[u].transactions[1], coinbase=True
                ),
                deltas,
                -1,
            )
            for tx in self.blockset[u].transactions[2:]:
                dt = self.get_transaction_deltas(tx, coinbase=False)
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
                        self._log("info", "Requested last parent to find LCA")
                        return

                    valid_chain = True
                    for in_block in path:
                        if valid_chain:
                            # undo coinbase transaction
                            self.apply_transaction(
                                self.get_transaction_deltas(
                                    self.blockset[in_block].transactions[0],
                                    coinbase=True,
                                ),
                                deltas,
                            )
                            self.apply_transaction(
                                self.get_transaction_deltas(
                                    self.blockset[in_block].transactions[1],
                                    coinbase=True,
                                ),
                                deltas,
                            )

                            for tx in self.blockset[in_block].transactions[2:]:
                                dt = self.get_transaction_deltas(tx, coinbase=False)
                                if all(
                                    self.chainstate[addr] + value + deltas[addr] >= 0
                                    for addr, value in dt.items()
                                ):
                                    self.apply_transaction(dt, deltas)
                                else:
                                    self._log("warn", "Invalid transaction")
                                    # TODO: mark block as invalid
                                    self.invalid_blocks.add(in_block)
                                    self.blockset.pop(in_block)
                                    valid_chain = False
                                    break
                        else:
                            self.invalid_blocks.add(in_block)
                            self.blockset.pop(in_block)
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

                        self._log("info", f"Current chainstate: {[f'{k} {v}' for k, v in self.chainstate.items()]}")
                        self._log("info", "Chain reorganization completed")

    def build_coinbase_transaction(self, multiplier: int = 1, token: int = 1) -> Transaction:
        output = [TxOut(self.my_peer.mid, BLOCK_REWARD * multiplier)]
        return self.make_and_sign_transaction(output, token)

    def apply_transaction(
        self, deltas: dict[tuple[bytes, int], int], chain_state: dict[tuple[bytes, int], int], sign: int = 1
    ):
        for address, value in deltas.items():
            chain_state[address] += sign * value

    def build_valid_transactions_for_block(
        self,
    ) -> tuple[list[Transaction], list[Transaction], defaultdict[[bytes,int], int]]:
        """
                Get valid transactions from mempool
                Delete invalid transactions from mempool while iterating
                Returns:
                    list[Transaction]: List of valid transactions for the next block
                """
        valid_transactions = []
        tx_to_remove = []
        deltas = defaultdict(int)
        for tx in self.mempool.values():
            if len(valid_transactions) >= BLOCK_SIZE - 1:
                break
            dt = self.get_transaction_deltas(tx, coinbase=False)
            if all(
                self.chainstate[addr] + value + deltas[addr] >= 0
                for addr, value in dt.items()
            ):
                valid_transactions.append(tx)
                self.apply_transaction(dt, deltas)
            tx_to_remove.append(tx)
        return valid_transactions, tx_to_remove, deltas

    def build_and_mine_block(self) -> LocalBlockOperation:
        # Extract transactions
        with self.lock_mining:
            transactions = [self.build_coinbase_transaction(), self.build_coinbase_transaction(token=Token['SHIBA_INU'], multiplier=2)]
            valid_txs, tx_to_remove, deltas = self.build_valid_transactions_for_block()
            transactions.extend(valid_txs)
            deltas[self.my_peer.mid,Token['NEPHERITE']] += BLOCK_REWARD
            deltas[self.my_peer.mid,Token['SHIBA_INU']] += 2 * BLOCK_REWARD

        tree = MerkleTree([tx.sign for tx in transactions])
        header = BlockHeader(
            seq_num=self.current_seq_num + 1,
            prev_block_hash=self.current_block_hash,
            merkle_root_hash=tree.root.hash,
            timestamp=time.monotonic_ns() // 1_000,
            difficulty=BLOCK_DIFFICULTY,
            nonce=0,
        )

        # Compute nonce
        block_hash = self.get_block_hash(header)
        nonce = Puzzle.compute(block_hash)
        header.nonce = nonce  # update nonce

        block_hash = self.get_block_hash(header)  # updated block hash
        block = Block(header, transactions)
        return (block, tx_to_remove, deltas)
