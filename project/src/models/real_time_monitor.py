import time
import torch
import numpy as np
from collections import deque

class RealTimeMonitor:
    def __init__(self, classifier, window_size=100):
        self.classifier = classifier
        self.window_size = window_size
        self.detection_history = deque(maxlen=window_size)
        self.last_update = time.time()
        
    def process_stream(self, data_batch):
        with torch.no_grad():
            logits, probs = self.classifier(torch.FloatTensor(data_batch))
            predictions = torch.argmax(probs, dim=1)
            warnings = self.classifier.check_warning(probs)
            
            # Update detection history
            self.detection_history.extend(predictions.numpy())
            
            # Check if model update is needed
            if time.time() - self.last_update > 3600:  # Update every hour
                self.update_model()
                
            return predictions.numpy(), warnings.numpy()
    
    def update_model(self):
        # Implement continuous learning logic here
        self.last_update = time.time()
        
    def get_statistics(self):
        if len(self.detection_history) == 0:
            return {}
            
        return {
            'total_detections': len(self.detection_history),
            'attack_ratio': np.mean([x != 0 for x in self.detection_history]),
            'unique_attacks': len(set(self.detection_history))
        }