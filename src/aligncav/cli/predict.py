"""
CLI command for mode prediction/inference.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


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
        description="Predict beam modes from images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input arguments
    parser.add_argument(
        "input",
        type=str,
        help="Input image file or directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )

    # Model arguments
    parser.add_argument(
        "--model-type",
        type=str,
        default="standard",
        choices=["standard", "deep"],
        help="Model architecture",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Input image size",
    )
    parser.add_argument(
        "--max-mode",
        type=int,
        default=10,
        help="Maximum mode index",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for predictions (CSV)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Show top-k predictions",
    )

    # Other arguments
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def load_and_preprocess_image(
    path: Path,
    image_size: int = 256,
) -> np.ndarray:
    """Load and preprocess an image."""
    import cv2

    # Load image
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {path}")

    # Resize
    img = cv2.resize(img, (image_size, image_size))

    # Normalize
    img = img.astype(np.float32) / 255.0

    return img


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    try:
        import torch
        import torch.nn.functional as F

        from ..models import DeepModeClassifier, ModeClassifier

        # Determine device
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Using device: {device}")

        # Calculate number of classes
        num_classes = (args.max_mode + 1) ** 2

        # Create model
        if args.model_type == "standard":
            model = ModeClassifier(
                num_classes=num_classes,
                input_size=args.image_size,
            )
        else:
            model = DeepModeClassifier(
                num_classes=num_classes,
                input_size=args.image_size,
            )

        # Load checkpoint
        checkpoint = torch.load(args.model, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()

        logger.info(f"Loaded model from {args.model}")

        # Get input files
        input_path = Path(args.input)
        if input_path.is_file():
            image_files = [input_path]
        elif input_path.is_dir():
            image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg"))
        else:
            logger.error(f"Invalid input: {args.input}")
            return 1

        if not image_files:
            logger.error("No image files found")
            return 1

        logger.info(f"Processing {len(image_files)} images")

        # Process images
        results: List[Tuple[str, int, int, float]] = []

        with torch.no_grad():
            for img_path in image_files:
                # Load and preprocess
                img = load_and_preprocess_image(img_path, args.image_size)
                tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

                # Predict
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)

                # Get top-k predictions
                top_probs, top_classes = probs.topk(args.top_k, dim=1)

                # Decode best prediction
                best_class = int(top_classes[0, 0].item())
                best_prob = float(top_probs[0, 0].item())
                m = best_class // (args.max_mode + 1)
                n = best_class % (args.max_mode + 1)

                results.append((img_path.name, m, n, best_prob))

                if args.verbose:
                    print(f"\n{img_path.name}:")
                    for i in range(args.top_k):
                        c = int(top_classes[0, i].item())
                        p = float(top_probs[0, i].item())
                        mi = c // (args.max_mode + 1)
                        ni = c % (args.max_mode + 1)
                        print(f"  TEM{mi}{ni}: {p:.4f}")
                else:
                    print(f"{img_path.name}: TEM{m}{n} ({best_prob:.4f})")

        # Save results if output specified
        if args.output:
            import csv

            with open(args.output, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "m", "n", "confidence"])
                writer.writerows(results)

            logger.info(f"Results saved to {args.output}")

        return 0

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
