"""Train ML models and optionally the small CNN."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train_ml import train_ml

def main():
    parser = argparse.ArgumentParser(description="Train models using your local HAM10000 dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of images to sample.")
    parser.add_argument("--cnn", action="store_true", help="Also train the TensorFlow CNN (takes longer).")
    parser.add_argument("--epochs", type=int, default=5, help="CNN epochs when --cnn is supplied.")
    args = parser.parse_args()
    if args.limit is None:
        from config import DEFAULT_TRAIN_LIMIT
        args.limit = DEFAULT_TRAIN_LIMIT
    print("Training SVM and Random Forest...")
    print(json.dumps(train_ml(args.limit), indent=2))
    if args.cnn:
        from src.train_cnn import train_cnn
        print("Training CNN. CPU training may take a while...")
        print(json.dumps(train_cnn(args.limit, args.epochs), indent=2))
if __name__ == "__main__":
    main()
