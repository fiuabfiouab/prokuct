import torch
from models.autoencoder import AutoEncoder
from models.attention import BAMAttention
from models.classifier import NetworkClassifier
from models.gan import SNGAN
from models.real_time_monitor import RealTimeMonitor
from utils.data_processor import DataProcessor
from utils.optimizer import PSOOptimizer

def main():
    # Initialize components
    data_processor = DataProcessor()
    
    # Load and preprocess data
    data = data_processor.load_dataset("CSE-CIC-IDS2018")
    processed_data = data_processor.preprocess(data)
    
    # Initialize models
    autoencoder = AutoEncoder(input_dim=80, encoding_dim=32)
    attention = BAMAttention(channels=32)
    gan = SNGAN(latent_dim=32)
    classifier = NetworkClassifier(num_classes=8)
    monitor = RealTimeMonitor(classifier)
    
    # Train pipeline
    print("Training representation learning pipeline...")
    encoded_data = autoencoder.encode(processed_data)
    attended_data = attention.forward(encoded_data)
    gan.train(attended_data)
    
    # Optimize classifier using PSO
    print("Optimizing classifier parameters...")
    pso = PSOOptimizer(n_particles=20, n_dimensions=32)
    best_params = pso.optimize(lambda x: -classifier.train(x, processed_data))
    
    # Start real-time monitoring
    print("Starting real-time monitoring...")
    while True:
        # Simulate real-time data stream
        new_data = data_processor.get_next_batch()
        predictions, warnings = monitor.process_stream(new_data)
        
        # Handle warnings
        if warnings.any():
            print("⚠️ Potential intrusion detected!")
            stats = monitor.get_statistics()
            print(f"Current statistics: {stats}")
        
        # Update model if needed
        if monitor.needs_update():
            monitor.update_model()

if __name__ == "__main__":
    main()