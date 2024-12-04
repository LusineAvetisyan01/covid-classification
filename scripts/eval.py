import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
import os

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model (adjust path as necessary)
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Adjust for binary classification
model.load_state_dict(torch.load("C:/Users/lusin/Desktop/covid_classifier/models/covid_classifier.pth"))
model.eval()  # Set the model to evaluation mode
model = model.to(device)

# Define transforms (same as used during training)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Same normalization as during training
])

# Load the evaluation dataset
data_dir = "C:/Users/lusin/Desktop/covid_classifier/data/eval"  # Directory containing eval images (non-COVID and COVID)
image_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

# DataLoader to load images in batches
eval_loader = DataLoader(image_dataset, batch_size=32, shuffle=False)

# Initialize lists to keep track of true labels and predicted labels
true_labels = []
predicted_labels = []

# Loop through the evaluation set
for inputs, labels in eval_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

    # Append the true labels and predicted labels
    true_labels.extend(labels.cpu().numpy())
    predicted_labels.extend(preds.cpu().numpy())

# Calculate Precision, Recall, and F1-score
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

# Print the results
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
