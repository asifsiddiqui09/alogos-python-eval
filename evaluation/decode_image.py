"""
Decode an image to raw RGBA bytes so the SAME pixel data can be fed to both
the Python and JavaScript detectors. This isolates the detection algorithm
from differences in image-decoding libraries (forensic accuracy).

Output: a .bin file (raw RGBA bytes) + a .json sidecar with width/height.
"""
import json
import sys
from pathlib import Path
from PIL import Image


def decode_to_rgba(image_path: str, out_path: str, max_size: int = 0):
    img = Image.open(image_path).convert("RGBA")

    # Optional downscale (speed only). Keeps aspect ratio.
    if max_size and max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.BILINEAR)

    width, height = img.size
    raw = img.tobytes()  # RGBA, 4 bytes per pixel

    out = Path(out_path)
    out.write_bytes(raw)
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps({
        "width": width,
        "height": height,
        "source": str(image_path),
        "bytes": len(raw),
    }, indent=2))

    print(f"Decoded {image_path} -> {width}x{height} ({len(raw)} bytes)")
    print(f"  raw : {out}")
    print(f"  meta: {meta_path}")
    return width, height


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python decode_image.py <image> <out.bin> [max_size]")
        sys.exit(1)
    max_size = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    decode_to_rgba(sys.argv[1], sys.argv[2], max_size)
