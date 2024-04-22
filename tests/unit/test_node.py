import os
import shutil
from collections.abc import Coroutine
from typing import Any

from ipv8.test.base import TestBase

import nepherite.puzzle
from nepherite.node import (
    Block,
    NepheriteNode,
    PullBlockRequest,
    Transaction,
    TxOut,
)


class NepheriteNodeTests(TestBase[NepheriteNode]):
    def setUp(self) -> None:
        super().setUp()

        nepherite.puzzle.DIFFICULTY = 2

        if not os.path.isdir("data/blocks"):
            os.mkdir("data/blocks")
        if not os.path.isdir("data/keys"):
            os.mkdir("data/keys")

    async def tearDown(self) -> Coroutine[Any, Any, None]:
        await super().tearDown()

        shutil.rmtree("data/keys", ignore_errors=True)
        shutil.rmtree("data/blocks", ignore_errors=True)

    async def test_genesis_block_exists(self):
        """
        Test genesis block is being created and stored in the blockset.
        """
        self.initialize(NepheriteNode, 2)
        node = self.overlay(0)

        hash_ = node.genesis_block_hash
        self.assertTrue(hash_ is not None)

        genesis_block = node.blockset.get(hash_)
        self.assertTrue(genesis_block is not None)
        self.assertEqual(genesis_block.header.seq_num, 0)
        self.assertEqual(genesis_block.header.prev_block_hash, b"\x00" * 32)
        self.assertEqual(genesis_block.header.timestamp, 0)

    async def test_node_messages(self):
        """
        Test that 2 nodes started correctly and are able to send messages to eachother.
        """
        self.initialize(NepheriteNode, 2)

        await self.introduce_nodes()

        peer_0 = self.nodes[0].my_peer
        peer_1 = self.nodes[1].my_peer

        # Check if the nodes are connected to eachother
        self.assertTrue(self.overlay(0).get_peers()[0].mid == peer_1.mid)
        self.assertTrue(self.overlay(1).get_peers()[0].mid == peer_0.mid)

        # Arrange a transaction
        tx_out = TxOut(peer_0.mid, 75)
        tx = self.overlay(0).make_and_sign_transaction([tx_out])

        # Send the transaction from node 0 to node 1 and assert that it was received
        with self.assertReceivedBy(1, [Transaction]) as received_messages:
            self.overlay(0).ez_send(peer_1, tx)
            await self.deliver_messages()

        # Assert that the received transaction is the same as the sent transaction
        self.assertEqual(tx, received_messages[0])

        # Arrange a block
        (block, _, _) = self.overlay(0).build_and_mine_block()

        # Send the block from node 0 to node 1 and assert that it was received
        with self.assertReceivedBy(1, [Block]) as received_messages:
            self.overlay(0).ez_send(peer_1, block)
            await self.deliver_messages()

        # Assert that the received block is the same as the sent block
        self.assertEqual(block, received_messages[0])

        # Arrange a pull block request
        pull_block_request = PullBlockRequest(block.header.merkle_root_hash)

        # Send the pull block request from node 0 to node 1 and assert that it was received
        with self.assertReceivedBy(1, [PullBlockRequest]) as received_messages:
            self.overlay(0).ez_send(peer_1, pull_block_request)
            await self.deliver_messages()

        # Assert that the received pull block request is the same as the sent pull block request
        self.assertEqual(pull_block_request, received_messages[0])

    async def test_node_methods(self):
        self.initialize(NepheriteNode, 2)

        await self.introduce_nodes()

        for i in range(2):
            self.overlay(i).start_to_create_block()

        # # Test the create_dummy_transaction method
        with self.assertReceivedBy(
            1, [Transaction, Block], ordered=False
        ) as received_messages:
            self.overlay(0).create_dummy_transaction()
            await self.deliver_messages()

        # Assert that peer_1 received the dummy transaction and that it is signed and from the correct address
        transaction = (
            received_messages[0]
            if received_messages[0].__class__ == Transaction
            else received_messages[1]
        )
        pk = self.overlay(0).crypto.key_from_public_bin(transaction.pk)
        addr = pk.key_to_hash()
        blob = self.overlay(0).serializer.pack_serializable(transaction.payload)
        self.assertTrue(
            self.overlay(0).crypto.is_valid_signature(pk, blob, transaction.sign)
        )
        self.assertEqual(addr, self.overlay(0).my_peer.mid)

    async def test_invalid_transaction(self):
        self.initialize(NepheriteNode, 2)

        await self.introduce_nodes()

        # Arrange a transaction
        tx_out = TxOut(self.nodes[0].my_peer.mid, 15)
        tx = self.overlay(0).make_and_sign_transaction([tx_out])
        # Alter the transaction payload
        tx.payload.output[0].amount = 50

        # Assert that the transaction is invalid
        self.assertFalse(self.overlay(0).verify_sign_transaction(tx))
