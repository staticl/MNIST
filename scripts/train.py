import numpy as np

import os
from pathlib import Path

from src.train.train import evalute, cnn_init
from src.train.plot import plot_training_loss, plot_val_metrics


PROCESSED_PATH = os.path.join(Path(r"data"), Path(r"mnist_preprocessed.npz"))

SEED = 101

EPOCHS = 1

CONV_LAYER_PARAMS = [(16, 1, 6), (32, 16, 8), (64, 32, 10)]

MODEL = Path("mnistCNN_model.npz")
SAVE = True


def main():
    cnn_componentes = cnn_init(PROCESSED_PATH, CONV_LAYER_PARAMS, seed=SEED)

    train_loader = cnn_componentes["train_loader"]
    val_loader = cnn_componentes["val_loader"]

    model = cnn_componentes["model"]
    crossEntropyWithLogitsLoss = cnn_componentes["crossEntropyWithLogitsLoss"]

    optimizer = cnn_componentes["optimizer"]

    if os.path.exists(MODEL):
        model_parameters = np.load(MODEL, allow_pickle=True)

        saved_params = list(model_parameters["parameters"])
        for saved_param, (params, _) in zip(saved_params, model.parameters()):
            params[:] = saved_param

        optimizer.m = list(model_parameters["m"])
        optimizer.v = list(model_parameters["v"])
        optimizer.t = int(model_parameters["t"])

    n_batches = len(train_loader)

    print(f"Total number of batches: {n_batches}")

    training_losses = []
    val_metrics = {
        "loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        total_loss = 0.0
        i = 0
        for X, y in train_loader:
            logits = model(X)
            i += 1
            total_loss += crossEntropyWithLogitsLoss(logits, y)
            dout = crossEntropyWithLogitsLoss.backward()
            model.backward(dout)
            optimizer.step()
        training_losses.append(total_loss / n_batches)

        if not ((epoch + 1) % 5):
            metrics = evalute(val_loader, model, crossEntropyWithLogitsLoss)
            for metric, value in metrics:
                val_metrics[metric].append(value)

    plot_training_loss(training_losses)
    plot_val_metrics(val_metrics)

    if SAVE:
        parameters = []
        for param, _ in model.parameters():
            parameters.append(param)

        np.savez(MODEL,
                 parameters=np.array(parameters, dtype=object),
                 m=np.array(optimizer.m, dtype=object),
                 v=np.array(optimizer.v, dtype=object),
                 t=optimizer.t,
                 allow_pickle=True)


if __name__ == "__main__":
    main()
