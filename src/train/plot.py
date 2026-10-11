import matplotlib.pyplot as plt


def plot_val_metrics(metrics: dict, per_n_epochs: int = 5) -> None:
    fig, axis = plt.subplots(len(metrics.keys()), 1, figsize=(11.69, 8.27))
    fig.suptitle(f"Different Validation Metrics per [{per_n_epochs}] Epochs")
    for ax, (metric, values) in zip(axis, metrics.items()):
        ax.plot(values, marker="+", color="steelblue", label=f"average validation {metric} per [{per_n_epochs}] epochs")
        ax.set_ylabel(f"{metric}")
        ax.set_xlabel(f"epochs [{per_n_epochs}]")
        ax.grid()
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_training_loss(loss: list[float]) -> None:
    fig, axis = plt.subplots(figsize=(8, 6))
    fig.suptitle("Training Loss per Epoch")
    axis.plot(loss, marker="+", color="orange", label="average training loss per epoch")
    axis.set_ylabel("loss")
    axis.set_xlabel("epoch")
    axis.grid()
    axis.legend()

    plt.tight_layout()
    plt.show()
