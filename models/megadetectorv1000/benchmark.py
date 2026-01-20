#!/usr/bin/env python3
"""Benchmark MegaDetector endpoint performance."""

import requests
import time
import json
import sys
from pathlib import Path
import statistics


def benchmark_endpoint(url, image_paths):
    """Benchmark endpoint and return timings."""
    timings = []

    for i, img_path in enumerate(image_paths, 1):
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        start = time.time()
        try:
            resp = requests.post(url, data=img_bytes, timeout=60)
            resp.raise_for_status()
            elapsed = time.time() - start
            timings.append(elapsed)
            print(f"  {i}/{len(image_paths)}: {elapsed:.3f}s")
        except Exception as e:
            print(f"  {i}/{len(image_paths)}: FAILED - {e}")

    return timings


def main():
    if len(sys.argv) != 4:
        print("Usage: python benchmark.py <model_name> <url> <output_json>")
        sys.exit(1)

    model_name = sys.argv[1]
    url = sys.argv[2]
    output_file = sys.argv[3]

    # Get all images
    validation_dir = Path("./validation")
    image_paths = sorted(validation_dir.glob("*.jpg")) + sorted(
        validation_dir.glob("*.JPG")
    )

    if not image_paths:
        print(f"No images found in {validation_dir}")
        sys.exit(1)

    print(f"Benchmarking {model_name} with {len(image_paths)} images...")
    timings = benchmark_endpoint(url, image_paths)

    if not timings:
        print("No successful requests")
        sys.exit(1)

    # Calculate stats
    results = {
        "model": model_name,
        "num_images": len(timings),
        "mean": statistics.mean(timings),
        "median": statistics.median(timings),
        "p95": sorted(timings)[int(len(timings) * 0.95)],
        "min": min(timings),
        "max": max(timings),
        "timings": timings,
    }

    # Load existing results or create new
    try:
        with open(output_file, "r") as f:
            all_results = json.load(f)
    except FileNotFoundError:
        all_results = []

    all_results.append(results)

    # Save results
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print(
        f"Mean: {results['mean']:.3f}s, Median: {results['median']:.3f}s, P95: {results['p95']:.3f}s"
    )


if __name__ == "__main__":
    main()
