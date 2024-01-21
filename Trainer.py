import torch
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import itertools
import io
import sys
import contextlib
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve, auc
from scipy import interp
import copy

class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.training_losses = []
        self.testing_losses = []
        self.validation_losses = []
        
        self.validation_accuracies = []
        self.training_accuracies = []
        self.testing_accuracies = []

        self.val_loader = 1

    def train(self, train_loader, test_loader, val_ratio=0.1, num_epochs=10, early_stopping_patience=5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        self.model.to(device)
        
        total_train_samples = len(train_loader.dataset)
        
        val_samples = int(total_train_samples * val_ratio)
        train_samples = total_train_samples - val_samples

        train_subset, val_subset = random_split(train_loader.dataset, [train_samples, val_samples])

        train_loader = DataLoader(train_subset, batch_size=train_loader.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=train_loader.batch_size, shuffle=False)
        self.val_loader = val_loader
        
        scheduler = StepLR(self.optimizer, step_size=1, gamma=0.95)
        early_stopping_counter = 0
        min_val_loss = float('inf')

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.unsqueeze(1).float().to(device), labels.squeeze().long().to(device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                del images, labels
                
            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = self.evaluate(train_loader)
            self.training_losses.append(avg_train_loss)
            self.training_accuracies.append(train_accuracy)


            val_loss, val_accuracy = self.evaluate_loader(val_loader, calc_loss=True)[:2]
            self.validation_losses.append(val_loss)
            self.validation_accuracies.append(val_accuracy)
        
            test_loss, test_accuracy = self.evaluate_loader(test_loader, calc_loss=True)[:2]
            self.testing_losses.append(test_loss)
            self.testing_accuracies.append(test_accuracy)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping triggered")
                    break
    
            scheduler.step()
            print(f'Epoch {epoch + 1}/{num_epochs}, '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%, ')
        return self.evaluate_loader(val_loader, calc_loss=True)

    def evaluate_loader(self, loader, calc_loss=False):
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        total_loss = 0.0
        all_labels = []
        all_predictions = []
        all_probs = []
    
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.unsqueeze(1).float().to(device), labels.squeeze().long().to(device)
                outputs = self.model(images)
                if calc_loss:
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.tolist())
                all_predictions.extend(predicted.tolist())
    
        avg_loss = total_loss / len(loader) if calc_loss else None
        accuracy = 100 * np.sum(np.array(all_labels) == np.array(all_predictions)) / len(all_labels)
    
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average=None, zero_division=0
        )

        zero_precision_labels = [label for label, prec in enumerate(precision) if prec == 0]
        if zero_precision_labels:
            pass
            # print(f"Labels with zero precision: {zero_precision_labels}")
    
        return avg_loss, accuracy, precision, recall, f1
        
    def evaluate(self, loader):
        return self.evaluate_loader(loader)[1]

    @staticmethod
    def do_train(model,  train_loader, test_loader, lr = 0.001, val_ratio=0.1, num_epochs=20, weight_decay=1e-4, early_stopping_patience=5):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        trainer = Trainer(model, criterion, optimizer)
        trainer.train(train_loader, test_loader, val_ratio=val_ratio, num_epochs=num_epochs, early_stopping_patience=early_stopping_patience)
        train_accuracy = trainer.evaluate(train_loader)
        test_accuracy = trainer.evaluate(test_loader)
        print(f'Train Accuracy: {train_accuracy}%')
        print(f'Test Accuracy: {test_accuracy}%')
        return trainer
        
    @staticmethod
    def plot_learning_curves(trainer):
        epochs = range(1, len(trainer.training_losses) + 1)
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(epochs, trainer.training_losses, label='Training Loss')
        plt.plot(epochs, trainer.testing_losses, label='Validation Loss')
        plt.title('Training and Testing Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(epochs, trainer.training_accuracies, label='Training Accuracy')
        plt.plot(epochs, trainer.testing_accuracies, label='Testing Accuracy')
        plt.title('Training and Testing Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def hyperparameter_tuning(model, train_loader, test_loader, param_grid, num_epochs=20):
        best_val_loss = float('inf')
        best_params = {}
        best_model = None
    
        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
        for params in param_combinations:
            current_model = copy.deepcopy(model)
            optimizer = torch.optim.Adam(current_model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
            criterion = nn.CrossEntropyLoss()
            trainer = Trainer(current_model, criterion, optimizer)
    
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.train(train_loader, test_loader, val_ratio=params['val_ratio'], num_epochs=num_epochs)
    
            val_loss, val_accuracy = trainer.evaluate_loader(test_loader, calc_loss=True)[:2]
    
            print(f"Parameters: {params}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%")
    
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params
                best_model = current_model
    
        print(f"Best Parameters: {best_params}, Best Validation Loss: {best_val_loss}")
    
        return best_model, best_params
