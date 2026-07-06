import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DEFAULT_METRICS_PATH = 'model/gan_training_metrics.csv'
DEFAULT_PLOT_PATH = 'model/loss_curve.png'


METRIC_COLUMNS = [
    'epoch',
    'd_loss',
    'g_loss_total',
    'g_adv_raw',
    'surrogate_raw',
    'thickness_raw',
    'g_adv_weighted',
    'surrogate_weighted',
    'thickness_weighted',
    'real_score',
    'fake_score',
    'grad_norm',
    'w_adv',
    'w_surrogate',
    'w_thickness',
]


def read_metrics(path):
    rows = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != METRIC_COLUMNS:
            raise ValueError(f"Unexpected metric columns in {path}: {reader.fieldnames}")
        for row in reader:
            rows.append({key: float(value) for key, value in row.items()})
    if len(rows) == 0:
        raise ValueError(f"No metric rows found in {path}")
    return rows


def column(rows, name):
    return [row[name] for row in rows]


def plot_gan_metrics(metrics_path=DEFAULT_METRICS_PATH, output_path=DEFAULT_PLOT_PATH):
    rows = read_metrics(metrics_path)
    epochs = column(rows, 'epoch')

    fig, axes = plt.subplots(4, 1, figsize=(11, 18))
    fig.tight_layout(pad=5.0)

    axes[0].plot(epochs, column(rows, 'g_loss_total'), label='G Loss Total')
    axes[0].plot(epochs, column(rows, 'g_adv_weighted'), label='G Adv Weighted')
    axes[0].set_title('Generator Total and Weighted Adversarial Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, column(rows, 'surrogate_weighted'), label='Surrogate Weighted')
    axes[1].plot(epochs, column(rows, 'thickness_weighted'), label='Thickness Weighted')
    axes[1].set_title('Generator Weighted Auxiliary Loss Terms')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(epochs, column(rows, 'd_loss'), label='D Loss')
    axes[2].plot(epochs, column(rows, 'real_score'), label='Critic Real Score')
    axes[2].plot(epochs, column(rows, 'fake_score'), label='Critic Fake Score')
    axes[2].set_title('Critic Diagnostics')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Value')
    axes[2].legend()
    axes[2].grid(True)

    axes[3].plot(epochs, column(rows, 'grad_norm'), label='GP Norm', color='orange')
    axes[3].axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
    axes[3].set_title('Gradient Penalty Norm')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Norm')
    axes[3].legend()
    axes[3].grid(True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Training plots saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot GAN training metrics from CSV')
    parser.add_argument('--metrics', default=DEFAULT_METRICS_PATH, help='Input metrics CSV path')
    parser.add_argument('--output', default=DEFAULT_PLOT_PATH, help='Output plot path')
    args = parser.parse_args()
    plot_gan_metrics(args.metrics, args.output)


if __name__ == '__main__':
    main()
