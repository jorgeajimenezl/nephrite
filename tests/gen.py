import argparse

from jinja2 import Environment, PackageLoader, select_autoescape

env = Environment(
    loader=PackageLoader("tests", package_path="."),
    autoescape=select_autoescape(),
)


def main():
    parser = argparse.ArgumentParser(
        prog="Scale the docker-compose file",
        description="Scale the number of nodes",
        epilog="Designed for A27 Fundamentals and Design of Blockchain-based Systems",
    )
    parser.add_argument("num_nodes", type=int)
    parser.add_argument(
        "template_file", type=str, nargs="?", default="docker-compose-template.yml"
    )
    parser.add_argument(
        "topology_file", type=str, nargs="?", default="topologies/ring.yaml"
    )

    args = parser.parse_args()
    template = env.get_template(args.template_file)

    with open("deploy/docker-compose.yml", "w") as f:
        content = template.render(
            num_nodes=args.num_nodes, topology_path=args.topology_file
        )
        f.write(content)
        print("Output written to docker-compose.yml")


if __name__ == "__main__":
    main()
