from nepherite.api.serializers import (
    BlockResource,
    NodeStats,
    TransactionResource,
    TxOutResource,
)
from nepherite.node import NepheriteNode


def get_snapshot(node: NepheriteNode) -> NodeStats:
    return NodeStats(
        current_seq_num=node.current_seq_num,
        current_block_hash=node.current_block_hash.hex()[:6],
        mempool=len(node.mempool),
        blockset=len(node.blockset),
        invalid_blocks=len(node.invalid_blocks),
        chainstate=len(node.chainstate),
    )


def get_blocks(node: NepheriteNode) -> list[BlockResource]:
    return [
        BlockResource(
            seq_num=block.header.seq_num,
            hash=node.get_block_hash(block.header).hex(),
            prev_block_hash=block.header.prev_block_hash.hex()[:6],
            merkle_root_hash=block.header.merkle_root_hash.hex()[:6],
            timestamp=block.header.timestamp,
            difficulty=block.header.difficulty,
            nonce=block.header.nonce,
            transactions=[
                TransactionResource(
                    nonce=tx.payload.nonce,
                    public_key=tx.pk.hex(),
                    signature=tx.sign.hex(),
                    output=[
                        TxOutResource(
                            address=tx_out.address.hex(),
                            amount=tx_out.amount,
                        )
                        for tx_out in tx.payload.output
                    ],
                )
                for tx in block.transactions
            ],
        )
        for block in node.blockset.values()
    ]


def get_block_by_hash(node: NepheriteNode, block_hash: bytes) -> BlockResource:
    block = node.blockset[block_hash]
    return BlockResource(
        seq_num=block.header.seq_num,
        hash=node.get_block_hash(block.header).hex(),
        prev_block_hash=block.header.prev_block_hash.hex(),
        merkle_root_hash=block.header.merkle_root_hash.hex()[:6],
        timestamp=block.header.timestamp,
        difficulty=block.header.difficulty,
        nonce=block.header.nonce,
        transactions=[
            TransactionResource(
                nonce=tx.payload.nonce,
                public_key=tx.pk.hex(),
                signature=tx.sign.hex(),
                output=[
                    TxOutResource(
                        address=tx_out.address.hex(),
                        amount=tx_out.amount,
                    )
                    for tx_out in tx.payload.output
                ],
            )
            for tx in block.transactions
        ],
    )


def get_mempool(node: NepheriteNode) -> TransactionResource:
    return [
        TransactionResource(
            nonce=tx.payload.nonce,
            public_key=tx.pk.hex(),
            signature=tx.sign.hex(),
            output=[
                TxOutResource(
                    address=tx_out.address.hex(),
                    amount=tx_out.amount,
                )
                for tx_out in tx.payload.output
            ],
        )
        for tx in node.mempool.values()
    ]
