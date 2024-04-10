import random
from collections import defaultdict

from ipv8.community import CommunitySettings
from ipv8.messaging.payload_dataclass import overwrite_dataclass
from dataclasses import dataclass

from ipv8.types import Peer

from da_types import Blockchain, message_wrapper

# We are using a custom dataclass implementation.
dataclass = overwrite_dataclass(dataclass)


@dataclass(
    msg_id=1
)  # The value 1 identifies this message and must be unique per community.
class Transaction:
    sender: int
    receiver: int
    amount: int
    nonce: int = 1


class BlockchainNode(Blockchain):

    def __init__(self, settings: CommunitySettings) -> None:
        super().__init__(settings)
        self.counter = 1
        self.max_messages = 5
        self.executed_checks = 0

        self.pending_txs = []
        self.finalized_txs = []
        self.balances = defaultdict(lambda: 1000)

        self.add_message_handler(Transaction, self.on_transaction)

    def on_start(self):
        if self.node_id % 2 == 0:
            #  Run client
            self.start_client()
        else:
            # Run validator
            self.start_validator()

    def create_transaction(self):
        peer = random.choice([i for i in self.get_peers() if self.node_id_from_peer(i) % 2 == 1])
        peer_id = self.node_id_from_peer(peer)

        tx = Transaction(self.node_id,
                         peer_id,
                         10,
                         self.counter)
        self.counter += 1
        print(f'[Node {self.node_id}] Sending transaction {tx.nonce} to {self.node_id_from_peer(peer)}')
        self.ez_send(peer, tx)

        if self.counter > self.max_messages:
            self.cancel_pending_task("tx_create")
            self.stop()
            return

    def start_client(self):
        # Create transaction and send to random validator
        # Or put node_id
        self.register_task("tx_create",
                           self.create_transaction, delay=1,
                           interval=1)

    def start_validator(self):
        self.register_task("check_txs", self.check_transactions, delay=2, interval=1)

    def check_transactions(self):
        for tx in self.pending_txs:
            if self.balances[tx.sender] - tx.amount >= 0:
                self.balances[tx.sender] -= tx.amount
                self.balances[tx.receiver] += tx.amount
                self.pending_txs.remove(tx)
                self.finalized_txs.append(tx)

        self.executed_checks += 1

        if self.executed_checks > 10:
            self.cancel_pending_task("check_txs")
            print(self.balances)
            self.stop()

    @message_wrapper(Transaction)
    async def on_transaction(self, peer: Peer, payload: Transaction) -> None:

        # Add to pending transactions
        if (payload.sender, payload.nonce) not in [(tx.sender, tx.nonce) for tx in self.finalized_txs] and (
        payload.sender, payload.nonce) not in [(tx.sender, tx.nonce) for tx in self.pending_txs]:
            self.pending_txs.append(payload)

        # Gossip to other nodes
        for peer in [i for i in self.get_peers() if self.node_id_from_peer(i) % 2 == 1]:
            self.ez_send(peer, payload)
