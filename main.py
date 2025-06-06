import argparse
import yaml
from datagenerator import generate_simulated_data
from dataprocess import preprocess_tdoa
from evaluation import evaluate
from visualization import plot_results
from model import load_model
from config import parse_args, load_yaml_config

def main():
    args = parse_args()
    config = load_yaml_config(args.cfg)

    print("==> Generating data...")
    true_positions, tdoa_measurements, problem_mask = generate_simulated_data(
        enable_nlos=args.nlos,
        enable_multipath=args.multipath,
        enable_blockage=args.blockage
    )

    print("==> Preprocessing TDOA data...")
    if args.process:
        processed_tdoa = preprocess_tdoa(tdoa_measurements, problem_mask, strategy="adaptive")
    else:
        processed_tdoa = tdoa_measurements

    print(f"==> Estimating positions using {args.model}...")
    estimate_positions = load_model(args.model, config)
    estimated_positions = estimate_positions(processed_tdoa)

    print("==> Evaluating results...")
    metrics, per_point_error = evaluate(estimated_positions, true_positions)

    print("Overall Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f} meters")

    print("Per-point Errors:")
    for i, err in enumerate(per_point_error):
        print(f"Point {i}: error = {err:.3f} meters")

    print("==> Visualizing...")
    plot_results(true_positions, estimated_positions, title=f"TDOA Positioning Results ({args.model})")

if __name__ == "__main__":
    main()
