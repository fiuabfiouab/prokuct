import numpy as np
import torch
import torch.nn as nn
from torchvision.models import densenet121

class NetworkClassifier(nn.Module):
    def __init__(self, num_classes, input_dim=80, hidden_dim=128):
        super().__init__()
        self.num_classes = num_classes
        
        # DenseNet for feature extraction
        self.densenet = densenet121(pretrained=True)
        self.densenet.features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        
        # CNN layers
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True, bidirectional=True)
        
        # Self-attention
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8)
        
        # Output layer
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        
        # Warning threshold
        self.threshold = 0.8
        
    def forward(self, x):
        # CNN feature extraction
        x_conv = self.conv(x.transpose(1, 2))
        
        # LSTM processing
        lstm_out, _ = self.lstm(x_conv.transpose(1, 2))
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Classification
        logits = self.classifier(attn_out.mean(1))
        probs = torch.softmax(logits, dim=1)
        
        return logits, probs
    
    def check_warning(self, probs):
        max_probs = torch.max(probs, dim=1)[0]
        warnings = max_probs > self.threshold
        return warnings
    
    def update_threshold(self, val_probs, val_labels, percentile=95):
        correct_probs = []
        for probs, label in zip(val_probs, val_labels):
            if torch.argmax(probs) == label:
                correct_probs.append(torch.max(probs).item())
        self.threshold = np.percentile(correct_probs, percentile)