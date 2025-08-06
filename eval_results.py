import pandas as pd
import matplotlib.pyplot as plt
import os

LOG_DIR = "results"

def plot_training_curve(exp_name):
    """Plot training metrics for a single experiment."""
    log_file = os.path.join(LOG_DIR, f"{exp_name}_training_log.csv")
    if not os.path.exists(log_file):
        print(f"No log file for {exp_name}")
        return

    df = pd.read_csv(log_file)
    plt.figure(figsize=(10,6))
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], marker="s", label="Val Loss")
    plt.plot(df["epoch"], df["precision"], marker="^", label="Precision")
    plt.plot(df["epoch"], df["recall"], marker="x", label="Recall")
    plt.plot(df["epoch"], df["f1"], marker="d", label="F1")

    plt.xlabel("Epoch")
    plt.ylabel("Score / Loss")
    plt.title(f"Training Curve - {exp_name}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    out_file = os.path.join(LOG_DIR, f"{exp_name}_training_curve.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Training curve saved to {out_file}")

def plot_f1_comparison(experiments):
    """Plot F1 curves of all experiments on the same figure."""
    plt.figure(figsize=(12,7))
    for exp in experiments:
        log_file = os.path.join(LOG_DIR, f"{exp}_training_log.csv")
        if not os.path.exists(log_file):
            print(f"Skipping {exp}, no log file.")
            continue
        df = pd.read_csv(log_file)
        plt.plot(df["epoch"], df["f1"], marker="o", label=exp)

    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 Score Comparison Across Experiments")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    out_file = os.path.join(LOG_DIR, "all_experiments_f1_comparison.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Overall F1 comparison plot saved to {out_file}")

if __name__ == "__main__":
    experiments = [
        "rgb_only_resnet18","ir_only_resnet18","rgb_only_mobilenet",
        "ir_only_mobilenet","fusion_early_resnet18","fusion_late_resnet18",
        "fusion_late_mobilenet","fusion_late_vgg16","fusion_late_efficientnet"
    ]

    for exp in experiments:
        plot_training_curve(exp)

    plot_f1_comparison(experiments)
