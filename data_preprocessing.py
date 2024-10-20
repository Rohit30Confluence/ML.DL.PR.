import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop("label", axis=1).values / 255.0  # Normalize pixel values
    y = data["label"].values
    return X, y

def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)

if __name__ == "__main__":
    X, y = load_data("data/mnist_train.csv")
    X_train, X_val, y_train, y_val = split_data(X, y)
    # Save the split data as needed for further use
