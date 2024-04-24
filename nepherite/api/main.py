import asyncio
import os
import sys
from collections.abc import Coroutine
from typing import Any

import utils
import uvicorn
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ipv8.configuration import (
    ConfigBuilder,
    Strategy,
    WalkerDefinition,
    default_bootstrap_defs,
)
from ipv8.util import create_event_with_signals
from ipv8_service import IPv8

from nepherite.node import NepheriteNode

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
)

ipv8_instance: IPv8 = None


async def run_blockchain(
    node_id: int,
    connections: list[int],
    use_localhost: bool = True,
    docker: bool = False,
) -> Coroutine[Any, None, None]:
    global ipv8_instance

    event = create_event_with_signals()

    builder = ConfigBuilder().clear_keys().clear_overlays()
    builder.add_key("nepherite-peer", "medium", f"data/keys/ec{node_id}.pem")
    if use_localhost:
        builder.set_port(9090 + node_id)
    builder.add_overlay(
        "blockchain_community",
        "nepherite-peer",
        (
            []
            if use_localhost
            else [WalkerDefinition(Strategy.RandomWalk, 10, {"timeout": 3.0})]
        ),
        default_bootstrap_defs,
        {},
        [("started", node_id, connections, event, use_localhost, docker)],
    )
    ipv8_instance = IPv8(
        builder.finalize(), extra_communities={"blockchain_community": NepheriteNode}
    )
    await ipv8_instance.start()
    await event.wait()
    await ipv8_instance.stop()


@app.get("/stats")
async def get_stats():
    node: NepheriteNode = ipv8_instance.overlays[0]
    with node.lock_mining:
        stats = utils.get_snapshot(node)
    return {
        "stats": stats,
    }


@app.get("/blocks")
async def get_blocks():
    node: NepheriteNode = ipv8_instance.overlays[0]
    with node.lock_mining:
        blocks = utils.get_blocks(node)
    return {
        "blocks": blocks,
    }


@app.get("/blocks/{block_hash}")
async def get_block(block_hash: str):
    node: NepheriteNode = ipv8_instance.overlays[0]
    with node.lock_mining:
        block = utils.get_block_by_hash(node, bytes.fromhex(block_hash))
    return {
        "block": block,
    }


@app.get("/mempool")
async def get_mempool():
    node: NepheriteNode = ipv8_instance.overlays[0]
    with node.lock_mining:
        mempool = utils.get_mempool(node)
    return {
        "mempool": mempool,
    }


def run_server(port: int):
    uvicorn.run(app, port=port)


async def main():
    os.makedirs("data/keys", exist_ok=True)
    os.makedirs("data/blocks", exist_ok=True)

    node_id = int(os.getenv("NODE_ID", 0))
    port = int(os.getenv("PORT", 8000))

    with open("tests/topologies/echo.yaml") as fd:
        topology = yaml.safe_load(fd)
        connections = topology[node_id]

    loop = asyncio.get_event_loop()

    await asyncio.gather(
        run_blockchain(node_id, connections),
        loop.run_in_executor(None, run_server, port),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
