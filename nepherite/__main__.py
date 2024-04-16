import argparse
import os
from asyncio import run

import yaml
from ipv8.configuration import ConfigBuilder, default_bootstrap_defs
from ipv8.util import create_event_with_signals
from ipv8_service import IPv8

from nepherite.node import NepheriteNode


async def start_communities(
    node_id: int,
    connections: list[int],
    use_localhost: bool = True,
) -> None:
    event = create_event_with_signals()
    base_port = 9090
    connections_updated = [(x, base_port + x) for x in connections]
    node_port = base_port + node_id
    builder = ConfigBuilder().clear_keys().clear_overlays()
    builder.add_key("nepherite-peer", "medium", f"data/keys/ec{node_id}.pem")
    builder.set_port(node_port)
    builder.add_overlay(
        "blockchain_community",
        "nepherite-peer",
        [],
        default_bootstrap_defs,
        {},
        [("started", node_id, connections_updated, event, use_localhost)],
    )
    ipv8_instance = IPv8(
        builder.finalize(), extra_communities={"blockchain_community": NepheriteNode}
    )
    await ipv8_instance.start()
    await event.wait()
    await ipv8_instance.stop()


if __name__ == "__main__":
    if not os.path.isdir("data"):
        os.mkdir("data")
    if not os.path.isdir("data/keys"):
        os.mkdir("data/keys")
    if not os.path.isdir("data/blocks"):
        os.mkdir("data/blocks")

    parser = argparse.ArgumentParser(
        prog="Blockchain",
        description="Code to execute blockchain.",
        epilog="Designed for A27 Fundamentals and Design of Blockchain-based Systems",
    )
    parser.add_argument("node_id", type=int)
    parser.add_argument(
        "topology", type=str, nargs="?", default="../tests/topologies/default.yaml"
    )
    parser.add_argument("--docker", action="store_true")
    args = parser.parse_args()
    node_id = args.node_id

    with open(args.topology) as f:
        topology = yaml.safe_load(f)
        connections = topology[node_id]

        run(start_communities(node_id, connections, not args.docker))
