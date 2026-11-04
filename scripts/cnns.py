"""One entrypoint for CNN data preparation, training, and evaluation."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()
    if args.command == "prepare":
        from protein_cnns.data import prepare_all
        result = prepare_all()
    elif args.command == "train":
        from protein_cnns.common import TASKS
        from protein_cnns.training import train_task
        result = {task: train_task(task, args.device, args.epochs, args.batch_size,
                                  args.learning_rate, args.weight_decay, args.patience)
                  for task in TASKS}
    else:
        from protein_cnns.evaluation import evaluate_all
        result, report = evaluate_all(args.device)
        print(f"report: {report}")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
