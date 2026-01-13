"""
CLI command for beam mode simulation and visualization.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and visualize beam modes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate mode images")
    gen_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for generated images",
    )
    gen_parser.add_argument(
        "--max-mode",
        type=int,
        default=10,
        help="Maximum mode index",
    )
    gen_parser.add_argument(
        "--samples-per-class",
        type=int,
        default=100,
        help="Samples per mode class",
    )
    gen_parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Output image size",
    )
    gen_parser.add_argument(
        "--variations",
        action="store_true",
        help="Include random variations",
    )

    # Visualize command
    vis_parser = subparsers.add_parser("visualize", help="Visualize beam modes")
    vis_parser.add_argument(
        "--m",
        type=int,
        default=0,
        help="Mode index m",
    )
    vis_parser.add_argument(
        "--n",
        type=int,
        default=0,
        help="Mode index n",
    )
    vis_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file (displays if not specified)",
    )
    vis_parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Image size",
    )

    # Grid command
    grid_parser = subparsers.add_parser("grid", help="Generate mode grid")
    grid_parser.add_argument(
        "--max-mode",
        type=int,
        default=5,
        help="Maximum mode index for grid",
    )
    grid_parser.add_argument(
        "--output",
        type=str,
        default="mode_grid.png",
        help="Output file",
    )
    grid_parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Size of each mode image",
    )

    # Common arguments
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate mode images."""
    import numpy as np
    from PIL import Image

    from ..simulation import HGModeGenerator, ModeParameters

    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = ModeParameters(image_size=args.image_size)
    generator = HGModeGenerator(params=params, max_mode=args.max_mode)

    total = 0
    for m in range(args.max_mode + 1):
        for n in range(args.max_mode + 1):
            class_idx = generator.get_class_from_indices(m, n)
            class_dir = output_dir / f"class_{class_idx}"
            class_dir.mkdir(exist_ok=True)

            for i in range(args.samples_per_class):
                if args.variations:
                    img = generator.generate_mode(
                        m,
                        n,
                        waist_scale=np.random.uniform(0.8, 1.2),
                        rotation=np.random.uniform(-0.1, 0.1),
                        offset=(np.random.uniform(-10, 10), np.random.uniform(-10, 10)),
                    )
                else:
                    img = generator.generate_mode(m, n)

                # Save as image
                img_uint8 = (img * 255).astype(np.uint8)
                Image.fromarray(img_uint8).save(class_dir / f"mode_{m}_{n}_{i:04d}.png")
                total += 1

            logger.info(f"Generated TEM{m}{n}: {args.samples_per_class} samples")

    logger.info(f"Total images generated: {total}")
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    """Visualize a single mode."""
    import matplotlib.pyplot as plt

    from ..simulation import HGModeGenerator, ModeParameters

    params = ModeParameters(image_size=args.image_size)
    generator = HGModeGenerator(params=params, add_noise=False)

    img = generator.generate_mode(args.m, args.n)

    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap="hot")
    plt.colorbar(label="Intensity")
    plt.title(f"TEM{args.m}{args.n} Mode")
    plt.xlabel("x (pixels)")
    plt.ylabel("y (pixels)")

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.output}")
    else:
        plt.show()

    return 0


def cmd_grid(args: argparse.Namespace) -> int:
    """Generate a grid of modes."""
    import matplotlib.pyplot as plt
    import numpy as np

    from ..simulation import HGModeGenerator, ModeParameters

    params = ModeParameters(image_size=args.image_size)
    generator = HGModeGenerator(params=params, add_noise=False)

    n_modes = args.max_mode + 1
    fig, axes = plt.subplots(n_modes, n_modes, figsize=(2 * n_modes, 2 * n_modes))

    for m in range(n_modes):
        for n in range(n_modes):
            img = generator.generate_mode(m, n)
            axes[m, n].imshow(img, cmap="hot")
            axes[m, n].set_title(f"({m},{n})", fontsize=8)
            axes[m, n].axis("off")

    plt.suptitle("Hermite-Gaussian Modes TEM$_{mn}$", fontsize=14)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved mode grid to {args.output}")

    return 0


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    if args.command == "generate":
        return cmd_generate(args)
    elif args.command == "visualize":
        return cmd_visualize(args)
    elif args.command == "grid":
        return cmd_grid(args)
    else:
        print("Please specify a command: generate, visualize, or grid")
        print("Use --help for more information")
        return 1


if __name__ == "__main__":
    sys.exit(main())
