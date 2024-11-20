import numpy as np

class DataProcessor:
    @staticmethod
    def load_dataset(path):
        # Simulated data loading for CSE-CIC-IDS2018
        print(f"Loading dataset from {path}")
        return np.random.rand(1000, 80)
    
    @staticmethod
    def preprocess(data):
        # Basic preprocessing steps
        return (data - data.mean()) / data.std()
    
    @staticmethod
    def split_data(X, y, test_size=0.2):
        n_samples = X.shape[0]
        n_test = int(n_samples * test_size)
        indices = np.random.permutation(n_samples)
        return (X[indices[n_test:]], y[indices[n_test:]],
                X[indices[:n_test]], y[indices[:n_test]])