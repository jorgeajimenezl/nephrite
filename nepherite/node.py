from dataclasses import dataclass

from cryptography.hazmat.primitives import hashes
from ipv8.community import CommunitySettings
from ipv8.messaging.payload_dataclass import dataclass as payload_dataclass
from ipv8.types import Peer
from rocksdict import Options, Rdict

from nepherite.base import Blockchain, message_wrapper


@dataclass
class Utxo:
    address: str
    amount: int


@payload_dataclass(msg_id=1)
class Transaction:
    txin: list[Utxo]
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


def sha256(data: bytes) -> bytes:
    digest = hashes.Hash(hashes.SHA3_256())
    digest.update(data)
    return digest.finalize()


class NepheriteNode(Blockchain):
    def __init__(self, settings: CommunitySettings) -> None:
        super().__init__(settings)

        self.parent: dict[bytes, bytes] = {}
        self.mempool: list[Transaction] = []

        options = Options(raw_mode=False)
        self.chainstate = Rdict("data/chainstate.db", options=options)

        self.add_message_handler(BlockHeader, self.on_block_header)
        self.add_message_handler(PullBlockRequest, self.on_pull_block_request)

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

    @message_wrapper(BlockHeader)
    def on_block_header(self, peer: Peer, header: BlockHeader) -> None:
        block_hash = self.get_block_hash(header)
        if block_hash in self.blocks:
            return
        
        # TODO: Update values and validate
        
    @message_wrapper(PullBlockRequest)
    def on_pull_block_request(self, peer: Peer, request: PullBlockRequest) -> None:
        block_hash = request.block_hash
        if block_hash not in self.blocks:
            self.parent[block_hash] = peer
            # broadcast pull request
            for u in self.get_peers():
                self.ez_send(u, request)
        block = self.blocks[block_hash]
        self.ez_send(peer, block)