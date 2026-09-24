"""
SEA-Biofouling-Quantification: Demonstration Script
Author: Research Team (Correspondence via manuscript contact)
License: Apache-2.0

This script demonstrates:
1. Instantiation and forward-pass verification of the YOLOv11-SEA architecture.
2. End-to-end evaluation using the indirect aperture-structure biofouling scoring engine.
3. Visualization of aperture-level blockage proportions and output generation.
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

from models.yolov11_sea import YOLOv11_SEA
from score.scorer import BiofoulingScorer, robust_read_image, robust_read_mask, robust_write_image


def generate_visualization(img_bgr: np.ndarray,
                           labeled_array: np.ndarray,
                           proportions: dict,
                           threshold: float,
                           norm_score: float) -> np.ndarray:
    """
    Generate an intuitive color overlay highlighting fouled apertures.
    Clean apertures (B_k <= T) are tinted green/cyan, while fouled apertures (B_k > T)
    are tinted yellow-to-red according to their excess severity.
    """
    h, w = img_bgr.shape[:2]
    overlay = img_bgr.copy().astype(np.float32)

    heatmap = np.zeros((h, w, 3), dtype=np.float32)
    mask_fouled = np.zeros((h, w), dtype=bool)

    for region_id, p_k in proportions.items():
        region_pixels = (labeled_array == region_id)
        if p_k > threshold:
            excess = min(1.0, (p_k - threshold) / (1.0 - threshold + 1e-6))
            # Yellow to bright red (BGR)
            b = 0.0
            g = float((1.0 - excess) * 200)
            r = 255.0
            heatmap[region_pixels] = [b, g, r]
            mask_fouled |= region_pixels
        else:
            # Subtle cyan/green tint for clean apertures
            heatmap[region_pixels] = [200.0, 200.0, 50.0]

    # Blend overlay
    alpha = 0.45
    blended = overlay.copy()
    blended[labeled_array > 0] = (
        (1 - alpha) * overlay[labeled_array > 0] + alpha * heatmap[labeled_array > 0]
    )
    result = np.clip(blended, 0, 255).astype(np.uint8)

    # Add HUD information bar at the top
    banner_height = 42
    banner = np.zeros((banner_height, w, 3), dtype=np.uint8)
    banner[:] = (35, 35, 35)

    info_text = f"Severity S_norm: {norm_score:.2f}% | Dynamic T: {threshold:.3f}"
    cv2.putText(banner, info_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    combined = np.vstack([banner, result])
    return combined


def run_demo(samples_dir: Path, output_dir: Path, verify_model: bool = True):
    """Run full demonstration pipeline on sample images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("SEA-Biofouling-Quantification: Minimal Reproducible Pipeline Demo")
    print("=" * 70)

    # 1. Model Architecture Verification
    if verify_model:
        print("\n[Step 1/3] Verifying YOLOv11-SEA Neural Network Definition...")
        model = YOLOv11_SEA(num_classes=1, in_channels=3)
        model.eval()
        dummy_input = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"  - Model initialized successfully.")
        print(f"  - Forward pass output shape: {list(output.shape)} (expected [1, 1, 640, 640])")

    # 2. Scorer Engine Initialization
    print("\n[Step 2/3] Initializing Biofouling Scoring Engine...")
    scorer = BiofoulingScorer()
    print(f"  - Successfully loaded {len(scorer.template_cache)} clean scale templates.")

    # 3. Processing Test Samples
    print("\n[Step 3/3] Quantifying Biofouling Severity on De-identified Samples...")
    sample_files = sorted(list(samples_dir.glob("*_clean.jpg")) +
                          list(samples_dir.glob("*_moderate.jpg")) +
                          list(samples_dir.glob("*_heavy.jpg")))

    if not sample_files:
        sample_files = sorted(list(samples_dir.glob("*.jpg")))

    if not sample_files:
        print(f"  [Warning] No test samples found in {samples_dir}.")
        return

    for img_path in sample_files:
        sample_stem = img_path.stem
        mask_path = img_path.with_name(f"{sample_stem}_mask.png")
        if not mask_path.exists():
            # Try alternate naming
            mask_path = img_path.with_name(f"{sample_stem}.png")
            if not mask_path.exists():
                print(f"  - Skipping {img_path.name}: mask file not found.")
                continue

        img_bgr = robust_read_image(img_path)
        mask = robust_read_mask(mask_path, target_size=(img_bgr.shape[0], img_bgr.shape[1]))

        # Calculate scores
        results = scorer.score_mask(mask)

        print(f"\n  -> Processing [{img_path.name}]:")
        print(f"     * Matched Template  : {results['matched_template']} (Dist: {results['distance_to_template']})")
        print(f"     * Dynamic Threshold : {results['threshold']:.4f} (Base: {results['base_threshold']:.4f} + Margin: {results['safety_margin']:.2f})")
        print(f"     * Aperture Count    : {results['num_fouled_apertures']} fouled / {results['num_total_apertures']} total")
        print(f"     * Raw Score S_raw   : {results['raw_score']:.1f} / S_max: {results['max_score']:.1f}")
        print(f"     * Normalized Score  : {results['normalized_score']:.2f}%")

        # Save visualization overlay
        viz = generate_visualization(
            img_bgr=img_bgr,
            labeled_array=results['labeled_array'],
            proportions=results['proportions'],
            threshold=results['threshold'],
            norm_score=results['normalized_score']
        )
        out_path = output_dir / f"result_{sample_stem}.png"
        robust_write_image(out_path, viz)
        print(f"     * Saved result to   : {out_path.name}")

    print("\n" + "=" * 70)
    print(f"Pipeline demonstration completed. All results saved to: {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="SEA-Biofouling-Quantification Demo")
    parser.add_argument("--samples_dir", type=str, default="samples", help="Path to samples directory")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Path to output directory")
    parser.add_argument("--skip_model", action="store_true", help="Skip model architecture forward test")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    samples_dir = project_root / args.samples_dir
    output_dir = project_root / args.output_dir

    run_demo(samples_dir=samples_dir, output_dir=output_dir, verify_model=not args.skip_model)


if __name__ == "__main__":
    main()
