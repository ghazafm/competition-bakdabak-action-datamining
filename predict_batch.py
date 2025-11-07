#!/usr/bin/env python
# coding: utf-8

"""
Enhanced prediction script with batch processing and confidence scores
"""

import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO


def predict_test_set_batch(
    model_path="model.pt",
    test_dir="data-mining-action-2025/test/test",
    output_csv="submission.csv",
    batch_size=16,
    img_size=640,
    save_confidence=False,
):
    """
    Load trained model, predict on test images with batch processing

    Args:
        model_path: Path to trained YOLO model
        test_dir: Directory containing test images
        output_csv: Output CSV filename
        batch_size: Number of images to process at once
        img_size: Image size for inference
        save_confidence: Whether to save confidence scores in separate file
    """
    # Convert to Path objects
    test_dir = Path(test_dir)

    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Check if test directory exists
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # Load the trained model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    # Set device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: cuda (GPU: {torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using device: mps (Apple Silicon)")
    else:
        device = "cpu"
        print("Using device: cpu")

    model.to(device)

    # Get all test images
    test_images = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")))
    print(f"Found {len(test_images)} test images")

    if len(test_images) == 0:
        raise ValueError("No test images found!")

    # Prepare results
    results_list = []

    # Process images in batches
    print(f"Starting predictions with batch size {batch_size}...")
    for i in tqdm(range(0, len(test_images), batch_size), desc="Batches"):
        batch_images = test_images[i : i + batch_size]
        batch_paths = [str(img) for img in batch_images]

        # Run batch prediction
        results = model.predict(
            source=batch_paths, imgsz=img_size, verbose=False, device=device
        )

        # Process results for each image in batch
        for img_path, result in zip(batch_images, results):
            # Get image ID from filename (e.g., "0001.jpg" -> 1)
            img_id = int(img_path.stem)

            # Get the predicted class name and confidence
            if hasattr(result, "probs"):
                # Classification model
                top1_idx = result.probs.top1
                predicted_class = result.names[top1_idx]
                confidence = float(result.probs.top1conf)
            else:
                # Detection model (fallback)
                if len(result.boxes) > 0:
                    top_box = result.boxes[0]
                    predicted_class = result.names[int(top_box.cls)]
                    confidence = float(top_box.conf)
                else:
                    # If no detection, use most common class as default
                    predicted_class = "Nasi Goreng"
                    confidence = 0.0

            result_dict = {"ID": img_id, "label": predicted_class}

            if save_confidence:
                result_dict["confidence"] = confidence

            results_list.append(result_dict)

    # Create DataFrame and sort by ID
    df = pd.DataFrame(results_list)
    df = df.sort_values("ID").reset_index(drop=True)

    # Save submission CSV (without confidence)
    submission_df = df[["ID", "label"]]
    submission_df.to_csv(output_csv, index=False)
    print(f"\n✓ Predictions saved to: {output_csv}")
    print(f"Total predictions: {len(df)}")

    # Save detailed results with confidence if requested
    if save_confidence:
        detailed_csv = output_csv.replace(".csv", "_detailed.csv")
        df.to_csv(detailed_csv, index=False)
        print(f"✓ Detailed predictions (with confidence) saved to: {detailed_csv}")
        print(f"\nAverage confidence: {df['confidence'].mean():.4f}")
        print(f"Min confidence: {df['confidence'].min():.4f}")
        print(f"Max confidence: {df['confidence'].max():.4f}")

    print("\nFirst 10 predictions:")
    print(submission_df.head(10))
    print("\nClass distribution:")
    print(df["label"].value_counts())

    # Check if predictions match expected count
    expected_count = len(
        [f for f in test_dir.iterdir() if f.suffix in [".jpg", ".png"]]
    )
    if len(df) != expected_count:
        print(
            f"\n⚠ Warning: Number of predictions ({len(df)}) doesn't match test images ({expected_count})"
        )

    return df


def verify_submission(
    csv_path, example_csv_path="data-mining-action-2025/example_test.csv"
):
    """
    Verify submission format matches the example

    Args:
        csv_path: Path to submission CSV
        example_csv_path: Path to example submission CSV
    """
    print("\nVerifying submission format...")

    # Load both files
    submission = pd.read_csv(csv_path)

    if os.path.exists(example_csv_path):
        example = pd.read_csv(example_csv_path)

        # Check columns
        if list(submission.columns) != list(example.columns):
            print("⚠ Warning: Columns don't match!")
            print(f"  Your columns: {list(submission.columns)}")
            print(f"  Expected: {list(example.columns)}")
        else:
            print("✓ Column names match")

        # Check number of rows
        if len(submission) != len(example):
            print("⚠ Warning: Row count doesn't match!")
            print(f"  Your rows: {len(submission)}")
            print(f"  Expected: {len(example)}")
        else:
            print("✓ Row count matches")

        # Check ID range
        if (
            submission["ID"].min() != example["ID"].min()
            or submission["ID"].max() != example["ID"].max()
        ):
            print("⚠ Warning: ID range doesn't match!")
            print(
                f"  Your ID range: {submission['ID'].min()} - {submission['ID'].max()}"
            )
            print(f"  Expected: {example['ID'].min()} - {example['ID'].max()}")
        else:
            print("✓ ID range matches")
    else:
        print(f"Example file not found at {example_csv_path}")

    # Check for missing IDs
    expected_ids = set(range(1, len(submission) + 1))
    actual_ids = set(submission["ID"])
    missing_ids = expected_ids - actual_ids

    if missing_ids:
        print(f"⚠ Warning: Missing IDs: {sorted(missing_ids)[:10]}...")
    else:
        print("✓ No missing IDs")

    # Check for duplicate IDs
    duplicate_ids = submission[submission.duplicated(subset=["ID"], keep=False)]
    if len(duplicate_ids) > 0:
        print(f"⚠ Warning: Duplicate IDs found: {duplicate_ids['ID'].unique()}")
    else:
        print("✓ No duplicate IDs")

    print("\nSubmission verification complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict test dataset for Kaggle submission"
    )
    parser.add_argument(
        "--model", type=str, default="model.pt", help="Path to trained model"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="data-mining-action-2025/test/test",
        help="Test images directory",
    )
    parser.add_argument(
        "--output", type=str, default="submission.csv", help="Output CSV filename"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for prediction"
    )
    parser.add_argument(
        "--img-size", type=int, default=640, help="Image size for inference"
    )
    parser.add_argument(
        "--save-confidence", action="store_true", help="Save confidence scores"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify submission format"
    )

    args = parser.parse_args()

    try:
        # Run predictions
        df = predict_test_set_batch(
            model_path=args.model,
            test_dir=args.test_dir,
            output_csv=args.output,
            batch_size=args.batch_size,
            img_size=args.img_size,
            save_confidence=args.save_confidence,
        )

        # Verify submission if requested
        if args.verify:
            verify_submission(args.output)

    except Exception as e:
        print(f"Error: {e}")
        raise
