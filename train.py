import numpy as np
from data_preprocessing import load_data, split_data
from model import create_model

def train_model():
    X, y = load_data("data/mnist_train.csv")
    X_train, X_val, y_train, y_val = split_data(X, y)

    model = create_model()
    model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))
    model.save('mnist_model.h5')

if __name__ == "__main__":
    train_model()
