import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta :
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Validation loss improved: {val_loss:.4f}")
        else :
            self.counter += 1
            if self.verbose:
                print(f"Validation loss did not improve: {val_loss:.4f}. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")

        return self.early_stop