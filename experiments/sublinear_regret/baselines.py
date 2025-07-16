"""
Baseline online learning algorithms for regret evaluation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional


class OnlineSGD:
    """Online Stochastic Gradient Descent baseline."""
    
    def __init__(self, input_dim: int, num_classes: int, learning_rate: float = 0.01):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
        # Initialize linear model
        self.model = nn.Linear(input_dim, num_classes)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make prediction for input x."""
        with torch.no_grad():
            return self.model(x)
    
    def update(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Update model with new sample and return loss."""
        # Ensure proper shapes
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0)
            
        # Forward pass
        logits = self.model(x)
        loss = self.criterion(logits, y.long())
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


class AdaGrad:
    """Adaptive Gradient Algorithm baseline."""
    
    def __init__(self, input_dim: int, num_classes: int, learning_rate: float = 0.01, eps: float = 1e-8):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.eps = eps
        
        # Initialize linear model
        self.model = nn.Linear(input_dim, num_classes)
        self.optimizer = optim.Adagrad(self.model.parameters(), lr=learning_rate, eps=eps)
        self.criterion = nn.CrossEntropyLoss()
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make prediction for input x."""
        with torch.no_grad():
            return self.model(x)
    
    def update(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Update model with new sample and return loss."""
        # Ensure proper shapes
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0)
            
        # Forward pass
        logits = self.model(x)
        loss = self.criterion(logits, y.long())
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


class OnlineNewtonStep:
    """Online Newton Step baseline (convex case)."""
    
    def __init__(self, input_dim: int, num_classes: int, learning_rate: float = 0.01, 
                 regularization: float = 1e-4):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.regularization = regularization
        
        # Initialize parameters
        self.W = torch.zeros(input_dim, num_classes, requires_grad=True)
        self.H = torch.eye(input_dim) * regularization  # Hessian approximation
        self.criterion = nn.CrossEntropyLoss()
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make prediction for input x."""
        with torch.no_grad():
            return torch.matmul(x, self.W)
    
    def update(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Update model with new sample and return loss."""
        # Ensure proper shapes
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 0:
            y = y.unsqueeze(0)
            
        # Forward pass
        logits = torch.matmul(x, self.W)
        loss = self.criterion(logits, y.long())
        
        # Compute gradients
        loss.backward()
        
        with torch.no_grad():
            # Get gradient
            grad = self.W.grad
            
            # Update Hessian approximation (diagonal approximation for efficiency)
            outer = torch.outer(x.flatten(), x.flatten())
            self.H += outer
            
            # Newton step update
            try:
                # Solve H * delta = -grad for delta
                H_inv = torch.inverse(self.H + self.regularization * torch.eye(self.input_dim))
                delta = -self.learning_rate * torch.matmul(H_inv, grad)
                self.W += delta
            except:
                # Fallback to gradient descent if Hessian is singular
                self.W -= self.learning_rate * grad
                
            # Clear gradients
            self.W.grad = None
            
        return loss.item()


def get_algorithm(algo_name: str, input_dim: int, num_classes: int, **kwargs):
    """Factory function to get algorithm instance."""
    if algo_name == "sgd":
        return OnlineSGD(input_dim, num_classes, **kwargs)
    elif algo_name == "adagrad":
        return AdaGrad(input_dim, num_classes, **kwargs)
    elif algo_name == "ons":
        return OnlineNewtonStep(input_dim, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")