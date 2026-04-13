from __future__ import annotations

import argparse
from python_cli.commands import handle_train, handle_predict, handle_evaluate, handle_clean


def _format_os_error(exc: OSError) -> str:
    path = getattr(exc, "filename", None)
    if path:
        return f"Cannot write to path: {path} ({exc.strerror or str(exc)})"
    return f"File system error: {exc}"

def check_dependencies():
    required_packages = ["numpy", "pandas", "sklearn", "matplotlib"]
    missing = []

    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("\n❌ Missing required Python packages:")
        for pkg in missing:
            print(f"  - {pkg}")

        print("\n👉 Run this to fix:")
        print("   pip install -r requirements.txt\n")
        raise SystemExit(1)

    try:
        __import__("xgboost")
    except ImportError:
        print("⚠️ Optional package 'xgboost' not found. XGBoost models will be skipped.")
        
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="MLPilot",
        description="MLPilot: Automated Model Selection and Evaluation Tool",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -------- TRAIN --------
    train_parser = subparsers.add_parser(
        "train",
        help="Train models and save the best one",
        description="Train models on a labeled dataset and save the best-performing model.",
    )
    train_parser.add_argument("--data", required=True, help="Path to training CSV file")
    train_parser.add_argument("--target", required=True, help="Target column name")
    train_parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split size between 0 and 1 (default: 0.2)",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    train_parser.add_argument(
        "--metric",
        default=None,
        help="Primary metric used to select best model",
    )
    train_parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save the trained model bundle",
    )
    train_parser.add_argument(
        "--output",
        default="outputs/output.json",
        help="Output JSON report path",
    )
    train_parser.add_argument(
        "--interface",
        default="python",
        choices=["python", "cpp", "bash", "java", "go"],
        help="Interface used to invoke the tool",
    )

    # -------- PREDICT --------
    predict_parser = subparsers.add_parser(
        "predict",
        help="Load a saved model and run inference",
        description="Run predictions on new data using a saved model.",
    )
    predict_parser.add_argument("--model", required=True, help="Path to saved model bundle")
    predict_parser.add_argument("--input", required=True, help="Path to input CSV file for inference")
    predict_parser.add_argument(
        "--output",
        default="outputs/predictions.csv",
        help="Path to save predictions CSV",
    )

    # -------- EVALUATE --------
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Load a saved model and evaluate on labeled data",
        description="Evaluate a saved model on a labeled dataset.",
    )
    evaluate_parser.add_argument("--model", required=True, help="Path to saved model bundle")
    evaluate_parser.add_argument("--data", required=True, help="Path to evaluation CSV file")
    evaluate_parser.add_argument("--target", required=True, help="Target column name")

    clean_parser = subparsers.add_parser(
        "clean",
        help="Remove all generated models and outputs",
        description="Deletes models/ and outputs/ directories.",
    )
    return parser.parse_args()

def main() -> None:
    check_dependencies()   # 👈 ADD THIS FIRST
    try:
        args = parse_args()

        if args.command == "train":
            handle_train(args, interface=args.interface)
        elif args.command == "predict":
            handle_predict(args)
        elif args.command == "evaluate":
            handle_evaluate(args)
        elif args.command == "clean":
            handle_clean()
        else:
            raise ValueError(f"Unsupported command: {args.command}")

    except FileNotFoundError as exc:
        print(f"Error: {exc}")
    except ValueError as exc:
        print(f"Input error: {exc}")
    except OSError as exc:
        print(f"Error: {_format_os_error(exc)}")
    except Exception as exc:
        print(f"Unexpected error: {exc}")

if __name__ == "__main__":
    main()
