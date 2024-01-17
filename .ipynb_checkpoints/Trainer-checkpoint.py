import torch
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader

class Trainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.training_losses = []
        self.testing_losses = []
        self.training_accuracies = []
        self.testing_accuracies = []

    def train(self, train_loader, test_loader, val_ratio=0.1, num_epochs=10):
        total_train_samples = len(train_loader.dataset)
        
        val_samples = int(total_train_samples * val_ratio)
        train_samples = total_train_samples - val_samples

        train_subset, val_subset = random_split(train_loader.dataset, [train_samples, val_samples])

        train_loader = DataLoader(train_subset, batch_size=train_loader.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=train_loader.batch_size, shuffle=False)

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.unsqueeze(1).float(), labels.squeeze().long()
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = self.evaluate(train_loader)
            self.training_losses.append(avg_train_loss)
            self.training_accuracies.append(train_accuracy)


            test_loss, test_accuracy = self.evaluate_loader(test_loader, calc_loss=True)
            self.testing_losses.append(test_loss)
            self.testing_accuracies.append(test_accuracy)
            
            print(f'Epoch {epoch + 1}/{num_epochs}, '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%, ')

    def evaluate_loader(self, loader, calc_loss=False):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.unsqueeze(1).float(), labels.squeeze().long()
                outputs = self.model(images)
                if calc_loss:
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_loss = total_loss / len(loader) if calc_loss else None
        accuracy = 100 * correct / total
        return avg_loss, accuracy
        
    def evaluate(self, data_loader):
        _, accuracy = self.evaluate_loader(data_loader)
        return accuracy

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
