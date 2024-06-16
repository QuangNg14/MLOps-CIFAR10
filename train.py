import os
import ssl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
import sys

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model definition
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_data(batch_size=64, subset_fraction=0.1):
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    print("Starting dataset download...")
    sys.stdout.flush()
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    print("Dataset download complete.")
    sys.stdout.flush()

    # Use a subset of the dataset
    train_size = int(subset_fraction * len(train_dataset))
    test_size = int(subset_fraction * len(test_dataset))
    train_dataset = Subset(
        train_dataset, np.random.choice(len(train_dataset), train_size, replace=False)
    )
    test_dataset = Subset(
        test_dataset, np.random.choice(len(test_dataset), test_size, replace=False)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print(f"Epoch [{epoch + 1}], Step [{i + 1}], Loss: {loss.item():.4f}")
            sys.stdout.flush()
    return running_loss / len(train_loader)


def validate(model, test_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss = val_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")
    sys.stdout.flush()
    return val_loss, accuracy


def fine_tune(model, train_loader, test_loader, epochs=5, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, epoch)
        val_loss, accuracy = validate(model, test_loader, criterion)

    return model


def main():
    print("Loading data...")
    sys.stdout.flush()
    train_loader, test_loader = load_data(subset_fraction=0.1)  # Use 10% of the dataset
    print("Data loading complete.")
    sys.stdout.flush()
    model = SimpleCNN().to(device)
    print("Starting training...")
    sys.stdout.flush()
    model = fine_tune(
        model, train_loader, test_loader, epochs=3, learning_rate=0.001
    )  # Reduce number of epochs to 3
    print("Training complete.")
    sys.stdout.flush()

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/simple_cnn.pth")
    print("Model saved to models/simple_cnn.pth")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
