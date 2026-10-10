import numpy as np

import os
from pathlib import Path 

from src.train.train import evalute, cnn_init
from src.train.plot import plot_training_loss, plot_val_metrics


PROCESSED_PATH = Path(r"data\mnist_preprocessed.npz")

SEED = 101

EPOCHS = 1

CONV_LAYER_PARAMS = [(16, 1, 6), (32, 16, 8), (64, 32, 10)]

MODEL = Path("mnistCNN_model.npz")

def main():
    cnn_componentes = cnn_init(PROCESSED_PATH, CONV_LAYER_PARAMS, seed=SEED)

    train_loader = cnn_componentes["train_loader"]
    val_loader = cnn_componentes["val_loader"]

    model = cnn_componentes["model"]
    crossEntropyWithLogitsLoss = cnn_componentes["crossEntropyWithLogitsLoss"]

    optimizer = cnn_componentes["optimizer"]
    
    if os.path.exists(MODEL):
        model_parameters = Path("mnistCNN_model.npz")

        saved_params = model_parameters["parameters"]
        for saved_param, (params, _) in zip(saved_params, model.parameters()):
            params[:] = saved_param

        optimizer.m = model_parameters["m"]
        optimizer.v = model_parameters["v"]
        optimizer.t = model_parameters["t"]

    n_batches = len(train_loader)

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
        for X, y in train_loader:
            logits = model(X)
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

if __name__ == "__main__":
    main()