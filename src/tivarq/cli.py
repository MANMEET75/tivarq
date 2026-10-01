"""TIVARQ command line interface."""
import argparse
import json
from pathlib import Path

from .dataset import validate_dataset, write_dataset
from .runner import run


def main():
    parser = argparse.ArgumentParser(prog="tivarq")
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("generate", "validate"):
        command = sub.add_parser(name)
        command.add_argument("--dataset", type=Path, default=Path("dataset"))
    command = sub.add_parser("run")
    command.add_argument("--dataset", type=Path, default=Path("dataset"))
    command.add_argument("--split", choices=("smoke", "dev", "test"), default="smoke")
    command.add_argument("--track", choices=("structured", "conversation"), default="structured")
    command.add_argument("--output", type=Path, default=Path("runs/smoke"))
    command.add_argument("--repeat", type=int, default=1)
    command.add_argument("--update-polls", type=int, default=0,
                         help="extra post-change polls for measuring asynchronous update lag")
    command.add_argument("--poll-interval-ms", type=int, default=100)
    command.add_argument("--seed", type=int, default=20261001)
    command.add_argument("--model-config", default="none")
    command.add_argument("--prompt-config", default="none")
    command.add_argument("adapter", nargs=argparse.REMAINDER,
                         help="adapter command after --, e.g. -- python -m tivarq.baselines versioned")
    args = parser.parse_args()
    if args.action == "generate":
        print(json.dumps(write_dataset(args.dataset), indent=2))
    elif args.action == "validate":
        print(json.dumps(validate_dataset(args.dataset), indent=2))
    else:
        cmd = args.adapter[1:] if args.adapter and args.adapter[0] == "--" else args.adapter
        if not cmd or args.repeat < 1 or args.update_polls < 0 or args.poll_interval_ms < 0:
            parser.error("run requires -- adapter-command and nonnegative polling settings")
        print(json.dumps(run(args.dataset, args.split, args.track, cmd, args.output,
            repeat=args.repeat, model_config=args.model_config, prompt_config=args.prompt_config,
            seed=args.seed, update_polls=args.update_polls,
            poll_interval_ms=args.poll_interval_ms)[0], indent=2))


if __name__ == "__main__":
    main()
