import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
}

# Load datasets
data_dir = "../data"
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms['train'])

# Dataloader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load ResNet-18
model = models.resnet18(pretrained=True)

# Modify the final layer for binary classification
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# Training function
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Training Loss: {epoch_loss:.4f}")

    return model


# Train the model
model = train_model(model, train_loader, criterion, optimizer, EPOCHS)

# Save the model
os.makedirs("../models", exist_ok=True)  # Ensure the directory exists
torch.save(model.state_dict(), "../models/covid_classifier.pth")
print("Model saved!")
