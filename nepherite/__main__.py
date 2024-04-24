import argparse
import asyncio

import yaml
from ipv8.configuration import (
    ConfigBuilder,
    Strategy,
    WalkerDefinition,
    default_bootstrap_defs,
)
from ipv8.util import create_event_with_signals
from ipv8_service import IPv8

from nepherite.base import BASE_PORT
from nepherite.node import NepheriteNode


async def start_communities(
    node_id: int,
    connections: list[int],
    use_localhost: bool = True,
    docker: bool = False,
) -> None:
    event = create_event_with_signals()

    builder = ConfigBuilder().clear_keys().clear_overlays()
    builder.add_key("nepherite-peer", "medium", f"data/keys/ec{node_id}.pem")
    if use_localhost:
        builder.set_port(BASE_PORT + node_id)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Blockchain",
        description="Code to execute blockchain.",
        epilog="Designed for A27 Fundamentals and Design of Blockchain-based Systems",
    )
    parser.add_argument(
        "--node_id",
        type=int,
        default=-1,
        help="Node ID to use in the topology on localhost",
    )
    parser.add_argument(
        "--topology", type=str, nargs="?", default="../tests/topologies/default.yaml"
    )
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--docker", action="store_true")
    args = parser.parse_args()

    node_id = args.node_id
    if args.local:
        with open(args.topology) as f:
            topology = yaml.safe_load(f)
            connections = topology[node_id]
    else:
        connections = []

    asyncio.run(start_communities(node_id, connections, args.local, args.docker))
