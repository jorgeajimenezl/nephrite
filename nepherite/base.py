from __future__ import annotations

import random
import traceback
from asyncio import Event
from collections.abc import Callable
from typing import Literal, TypeVar

from ipv8.community import Community, CommunitySettings
from ipv8.lazy_community import lazy_wrapper
from ipv8.messaging.serialization import Payload
from ipv8.types import LazyWrappedHandler, MessageHandlerFunction, Peer

from nepherite.utils import logging

DataclassPayload = TypeVar("DataclassPayload")
AnyPayload = Payload | DataclassPayload
BASE_PORT = 9090


def message_wrapper(
    *payloads: type[AnyPayload],
) -> Callable[[LazyWrappedHandler], MessageHandlerFunction]:
    return lazy_wrapper(*payloads)


class Blockchain(Community):
    community_id = b"matadores_ledgersuwu"

    def __init__(self, settings: CommunitySettings) -> None:
        super().__init__(settings)
        self.event: Event = None

    def _log(self, level: Literal["info", "warn", "error", "debug"], msg: str):
        logging.log(
            logging._nameToLevel[level.upper()],
            f"Node {self.my_peer.mid.hex()[:5]}: {msg}",
        )

        if level == "error":
            logging.error(traceback.format_exc())

    async def setup_localhost(self, node_id: int, connections: list[int]):
        self._log("info", "Setting up localhost")
        self._log("info", f"Node ID: {node_id}")

        connections: list[tuple[str, int]] = [(x, BASE_PORT + x) for x in connections]
        host_network, *_ = self._get_lan_address()

        async def _ensure_nodes_connected() -> None:
            for _, port in connections:
                addr = (host_network, port)
                self.walk_to(addr)
            valid = False
            conn_nodes = []

            for node_id, port in connections:
                conn_nodes = [p for p in self.get_peers() if p.address[1] == port]
                if len(conn_nodes) == 0:
                    return
                valid = True
                self._log("info", f"Store {conn_nodes[0]} with {node_id}")
            if not valid:
                return
            
            self._log("info", "Fully connected to all nodes in the topology")
            self.cancel_pending_task("ensure_nodes_connected")
            self._log("info", "Starting")

            delay = random.uniform(1.0, 3.0)
            self.register_anonymous_task("delayed_start", self.on_start, delay=delay)

        self.register_task(
            "ensure_nodes_connected", _ensure_nodes_connected, interval=0.5, delay=1
        )

    async def started(
        self,
        node_id: int,
        connections: list[tuple[int, int]],
        event: Event,
        use_localhost: bool = True,
    ) -> None:
        self._log("info", "Started!!")
        self.event = event

        if use_localhost:
            await self.setup_localhost(node_id, connections)
        else:
            self._log("info", "Not using localhost")
            self.on_start()

    def on_start(self):
        pass

    def stop(self, delay: int = 0):
        async def delayed_stop():
            self._log("info", "Stopping")
            self.event.set()

        self.register_anonymous_task("delayed_stop", delayed_stop, delay=delay)

    def ez_send(self, peer: Peer, *payloads: AnyPayload, **kwargs) -> None:
        super().ez_send(peer, *payloads, **kwargs)

    def add_message_handler(
        self, msg_num: int | type[AnyPayload], callback: MessageHandlerFunction
    ) -> None:
        super().add_message_handler(msg_num, callback)
