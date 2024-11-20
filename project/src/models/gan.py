import numpy as np

class SNGAN:
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        
    def generate(self, n_samples):
        return np.random.rand(n_samples, self.latent_dim)
        
    def discriminate(self, X):
        return np.random.rand(X.shape[0], 1)
        
    def train(self, X, epochs=10):
        print("Training SNGAN...")