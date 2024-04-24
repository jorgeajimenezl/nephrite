import asyncio
import os
import sys

import uvicorn
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from ipv8.configuration import (
    ConfigBuilder,
    Strategy,
    WalkerDefinition,
    default_bootstrap_defs,
)
from ipv8.util import create_event_with_signals
from ipv8_service import IPv8

import nepherite.api.utils as utils
from nepherite.node import NepheriteNode

load_dotenv()


ipv8_instance: IPv8 = None


async def run_blockchain(
    node_id: int,
    connections: list[int],
    use_localhost: bool = True,
    docker: bool = False,
):
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    node_id = int(os.getenv("NODE_ID", 0))

    with open("tests/topologies/echo.yaml") as fd:
        topology = yaml.safe_load(fd)
        connections = topology[node_id]

    loop = asyncio.get_event_loop()
    loop.create_task(run_blockchain(node_id, connections))
    yield
    loop.stop()
    await ipv8_instance.stop()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
)


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


if __name__ == "__main__":
    try:
        os.makedirs("data/keys", exist_ok=True)
        os.makedirs("data/blocks", exist_ok=True)

        node_id = int(os.getenv("NODE_ID", 0))
        port = int(os.getenv("PORT", 8000))

        with open("tests/topologies/echo.yaml") as fd:
            topology = yaml.safe_load(fd)
            connections = topology[node_id]

        run_server(port)
    except KeyboardInterrupt:
        sys.exit(0)
