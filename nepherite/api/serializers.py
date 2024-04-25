from ipv8.messaging.payload_dataclass import dataclass


@dataclass
class NodeStats:
    current_seq_num: int
    current_block_hash: bytes
    mempool: int
    blockset: int
    invalid_blocks: int
    chainstate: int


@dataclass
class TxOutResource:
    address: bytes
    amount: int


@dataclass
class TransactionResource:
    nonce: int
    public_key: bytes
    signature: bytes
    output: list[TxOutResource]


@dataclass
class BlockResource:
    seq_num: int
    hash: bytes
    prev_block_hash: bytes
    merkle_root_hash: bytes
    timestamp: int
    difficulty: int
    nonce: int
    transactions: list[TransactionResource]
