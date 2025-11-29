#!/usr/bin/env python
"""
Run MMM Platform Demo

This script demonstrates all the new features without needing a fitted model.

Usage:
    python run_demo.py                    # Run full demo
    python run_demo.py --summary-only     # Print summaries only (no visualizations)
    python run_demo.py --viz-only         # Show visualizations only
    python run_demo.py --save-viz PATH    # Save visualizations to PATH

Examples:
    python run_demo.py
    python run_demo.py --summary-only
    python run_demo.py --save-viz ./demo_output
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mmm_platform.analysis import (
    create_demo_scenario,
    run_full_demo,
    test_marginal_roi,
    test_executive_summary,
    test_combined_models,
    test_visualizations,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run MMM Platform Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print summaries only (no visualizations)"
    )
    parser.add_argument(
        "--viz-only",
        action="store_true",
        help="Show visualizations only (no summaries)"
    )
    parser.add_argument(
        "--save-viz",
        type=str,
        metavar="PATH",
        help="Save visualizations to PATH instead of displaying"
    )
    parser.add_argument(
        "--test",
        choices=["marginal", "executive", "combined", "viz"],
        help="Run specific test only"
    )
    parser.add_argument(
        "--weeks",
        type=int,
        default=104,
        help="Number of weeks of mock data (default: 104)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("MMM PLATFORM DEMO")
    print("=" * 80)
    print()

    # Run specific test if requested
    if args.test:
        if args.test == "marginal":
            print("Running Marginal ROI test...\n")
            test_marginal_roi()
        elif args.test == "executive":
            print("Running Executive Summary test...\n")
            test_executive_summary()
        elif args.test == "combined":
            print("Running Combined Models test...\n")
            test_combined_models()
        elif args.test == "viz":
            print("Running Visualizations test...\n")
            test_visualizations(save_path=args.save_viz)
        return

    # Create demo scenario
    demo = create_demo_scenario(
        n_weeks=args.weeks,
        seed=args.seed,
    )

    # Print summaries
    if not args.viz_only:
        demo.print_all_summaries()

        # Show DataFrames
        print("\n" + "=" * 80)
        print("AVAILABLE DATAFRAMES")
        print("=" * 80)

        dfs = demo.get_all_dataframes()
        for name, df in dfs.items():
            print(f"\n{name}:")
            print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
            if name == 'priority_table':
                print(df.to_string())

    # Show visualizations
    if not args.summary_only:
        print("\n" + "=" * 80)
        print("VISUALIZATIONS")
        print("=" * 80)

        if args.save_viz:
            print(f"\nSaving visualizations to: {args.save_viz}")
            demo.show_all_visualizations(save_path=args.save_viz)
            print(f"\nVisualizations saved to: {args.save_viz}")
        else:
            print("\nGenerating visualizations (close windows to continue)...")
            demo.show_all_visualizations()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
