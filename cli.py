"""Command-line entry point."""

from .config import parse_config
from .display import run_gif, run_live
from .simulator import Simulator
from .sources import Source


def main(argv=None):
    cfg = parse_config(argv)
    sim = Simulator(cfg)
    src = Source(cfg)
    try:
        (run_gif if cfg.save_gif else run_live)(cfg, sim, src)
    finally:
        src.release()


if __name__ == "__main__":
    main()
