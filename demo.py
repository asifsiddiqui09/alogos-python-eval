"""
Alogos Python Demo
Usage: python demo.py <image_path> [--threshold 0.6]
Requires: pip install Pillow
"""
import sys
import argparse
from alogos import detect_synthetic_image, ImageData, DetectorOptions


def load_image(path: str) -> ImageData:
    try:
        from PIL import Image
    except ImportError:
        print("Pillow is required to load images: pip install Pillow")
        sys.exit(1)

    img = Image.open(path).convert("RGBA")
    width, height = img.size
    data = list(img.tobytes())
    return ImageData(width=width, height=height, data=data)


def main():
    parser = argparse.ArgumentParser(description="Detect if an image is AI-generated")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--threshold", type=float, default=0.6, help="Detection threshold (default: 0.6)")
    args = parser.parse_args()

    print(f"\nLoading: {args.image}")
    image_data = load_image(args.image)
    print(f"Size: {image_data.width}x{image_data.height} ({image_data.width * image_data.height:,} pixels)")

    options = DetectorOptions(threshold=args.threshold)
    result = detect_synthetic_image(image_data, options)

    print("\n--- Detection Result ---")
    label = "AI-Generated (Synthetic)" if result.is_synthetic else "Real Photo"
    print(f"Verdict:          {label}")
    print(f"Raw score:        {result.raw_score:.4f}  (threshold: {args.threshold})")
    print(f"Confidence:       {result.confidence * 100:.1f}%")
    print(f"Primary variance: {result.metadata.primary_variance:.4f}")
    print(f"Coherence:        {result.metadata.coherence:.4f}")
    print(f"Pixels analysed:  {result.metadata.pixels_analysed:,}")


if __name__ == "__main__":
    main()
