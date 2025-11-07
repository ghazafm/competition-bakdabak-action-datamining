#!/usr/bin/env python
# coding: utf-8

"""
Utility script to analyze and validate predictions
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def analyze_predictions(csv_path="submission.csv", save_plots=True):
    """
    Analyze prediction results and generate visualizations

    Args:
        csv_path: Path to prediction CSV
        save_plots: Whether to save visualization plots
    """
    print(f"Analyzing predictions from: {csv_path}")

    # Load predictions
    df = pd.read_csv(csv_path)

    # Basic statistics
    print("\n" + "=" * 50)
    print("BASIC STATISTICS")
    print("=" * 50)
    print(f"Total predictions: {len(df)}")
    print(f"ID range: {df['ID'].min()} to {df['ID'].max()}")
    print(f"Number of unique classes: {df['label'].nunique()}")

    # Check data integrity
    print("\n" + "=" * 50)
    print("DATA INTEGRITY CHECKS")
    print("=" * 50)

    # Check for missing IDs
    expected_ids = set(range(df["ID"].min(), df["ID"].max() + 1))
    actual_ids = set(df["ID"])
    missing_ids = expected_ids - actual_ids

    if missing_ids:
        print(f"❌ Missing IDs: {len(missing_ids)}")
        print(f"   First few: {sorted(list(missing_ids))[:10]}")
    else:
        print("✅ No missing IDs")

    # Check for duplicate IDs
    duplicates = df[df.duplicated("ID", keep=False)]
    if len(duplicates) > 0:
        print(f"❌ Duplicate IDs found: {len(duplicates)}")
        print(duplicates.head())
    else:
        print("✅ No duplicate IDs")

    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        print("❌ Null values found:")
        print(null_counts[null_counts > 0])
    else:
        print("✅ No null values")

    # Class distribution
    print("\n" + "=" * 50)
    print("CLASS DISTRIBUTION")
    print("=" * 50)
    class_counts = df["label"].value_counts()
    print(class_counts)

    print(
        f"\nMost common: {class_counts.index[0]} ({class_counts.iloc[0]} predictions)"
    )
    print(
        f"Least common: {class_counts.index[-1]} ({class_counts.iloc[-1]} predictions)"
    )

    # Distribution statistics
    print(f"\nMean predictions per class: {class_counts.mean():.2f}")
    print(f"Std dev: {class_counts.std():.2f}")
    print(f"Min: {class_counts.min()}")
    print(f"Max: {class_counts.max()}")

    # Expected classes
    expected_classes = {
        "Ayam Bakar",
        "Ayam Betutu",
        "Ayam Goreng",
        "Ayam Pop",
        "Bakso",
        "Coto Makassar",
        "Gado Gado",
        "Gudeg",
        "Nasi Goreng",
        "Pempek",
        "Rawon",
        "Rendang",
        "Sate Madura",
        "Sate Padang",
        "Soto",
    }

    actual_classes = set(df["label"].unique())
    unexpected_classes = actual_classes - expected_classes
    missing_classes = expected_classes - actual_classes

    print("\n" + "=" * 50)
    print("CLASS VALIDATION")
    print("=" * 50)

    if unexpected_classes:
        print(f"⚠️  Unexpected classes found: {unexpected_classes}")
    else:
        print("✅ All classes are valid")

    if missing_classes:
        print(f"ℹ️  Classes not predicted: {missing_classes}")
    else:
        print("✅ All expected classes are present")

    # Create visualizations
    if save_plots:
        print("\n" + "=" * 50)
        print("GENERATING VISUALIZATIONS")
        print("=" * 50)

        # Set style
        sns.set_style("whitegrid")
        plt.figure(figsize=(14, 8))

        # Bar plot
        plt.subplot(2, 1, 1)
        class_counts.plot(kind="bar", color="steelblue")
        plt.title("Class Distribution in Predictions", fontsize=14, fontweight="bold")
        plt.xlabel("Food Class")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Add count labels on bars
        for i, v in enumerate(class_counts):
            plt.text(i, v + 1, str(v), ha="center", va="bottom", fontsize=9)

        # Pie chart for top classes
        plt.subplot(2, 1, 2)
        top_n = 10
        top_classes = class_counts.head(top_n)
        other_count = class_counts[top_n:].sum()

        if other_count > 0:
            plot_data = pd.concat([top_classes, pd.Series({"Others": other_count})])
        else:
            plot_data = top_classes

        colors = sns.color_palette("husl", len(plot_data))
        plt.pie(
            plot_data,
            labels=plot_data.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
        )
        plt.title(f"Top {top_n} Classes Distribution", fontsize=14, fontweight="bold")

        plt.tight_layout()

        # Save figure
        output_path = csv_path.replace(".csv", "_analysis.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ Visualization saved to: {output_path}")
        plt.close()

    return df


def compare_with_example(
    submission_csv="submission.csv",
    example_csv="data-mining-action-2025/example_test.csv",
):
    """
    Compare your submission with the example to check format

    Args:
        submission_csv: Your submission file
        example_csv: Example submission file
    """
    print("\n" + "=" * 50)
    print("COMPARING WITH EXAMPLE")
    print("=" * 50)

    if not Path(example_csv).exists():
        print(f"⚠️  Example file not found: {example_csv}")
        return

    sub = pd.read_csv(submission_csv)
    ex = pd.read_csv(example_csv)

    # Compare structure
    print("\nYour submission:")
    print(f"  Rows: {len(sub)}")
    print(f"  Columns: {list(sub.columns)}")
    print(f"  ID range: {sub['ID'].min()} - {sub['ID'].max()}")

    print("\nExample submission:")
    print(f"  Rows: {len(ex)}")
    print(f"  Columns: {list(ex.columns)}")
    print(f"  ID range: {ex['ID'].min()} - {ex['ID'].max()}")

    # Check if they match
    if len(sub) == len(ex):
        print("\n✅ Row count matches!")
    else:
        print(f"\n❌ Row count mismatch! Difference: {abs(len(sub) - len(ex))}")

    if list(sub.columns) == list(ex.columns):
        print("✅ Columns match!")
    else:
        print("❌ Columns don't match!")

    # Compare class distributions
    print("\nClass distribution comparison:")
    comparison = (
        pd.DataFrame(
            {
                "Your_Submission": sub["label"].value_counts(),
                "Example": ex["label"].value_counts(),
            }
        )
        .fillna(0)
        .astype(int)
    )

    comparison["Difference"] = comparison["Your_Submission"] - comparison["Example"]
    print(comparison)


def generate_summary_report(csv_path="submission.csv"):
    """
    Generate a comprehensive summary report

    Args:
        csv_path: Path to prediction CSV
    """
    df = pd.read_csv(csv_path)

    report = f"""
{"=" * 60}
PREDICTION SUMMARY REPORT
{"=" * 60}

Generated for: {csv_path}

OVERVIEW
--------
Total Predictions: {len(df)}
ID Range: {df["ID"].min()} - {df["ID"].max()}
Unique Classes: {df["label"].nunique()}

CLASS DISTRIBUTION
------------------
{df["label"].value_counts().to_string()}

STATISTICS
----------
Mean per class: {df["label"].value_counts().mean():.2f}
Std deviation: {df["label"].value_counts().std():.2f}
Min count: {df["label"].value_counts().min()}
Max count: {df["label"].value_counts().max()}

TOP 5 CLASSES
-------------
{df["label"].value_counts().head().to_string()}

BOTTOM 5 CLASSES
----------------
{df["label"].value_counts().tail().to_string()}

{"=" * 60}
"""

    # Save report
    report_path = csv_path.replace(".csv", "_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(report)
    print(f"\n✅ Report saved to: {report_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze prediction results")
    parser.add_argument(
        "--csv", type=str, default="submission.csv", help="Path to prediction CSV"
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument(
        "--compare", action="store_true", help="Compare with example submission"
    )
    parser.add_argument("--report", action="store_true", help="Generate summary report")

    args = parser.parse_args()

    try:
        # Analyze predictions
        df = analyze_predictions(args.csv, save_plots=not args.no_plots)

        # Compare with example if requested
        if args.compare:
            compare_with_example(args.csv)

        # Generate report if requested
        if args.report:
            generate_summary_report(args.csv)

    except Exception as e:
        print(f"Error: {e}")
        raise
