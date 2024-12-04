import torch
from torchvision import models, transforms
from PIL import Image
import os

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model (adjust path as necessary)
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Adjust for binary classification
model.load_state_dict(torch.load("C:/Users/lusin/Desktop/covid_classifier/models/covid_classifier.pth"))  
model.eval()  # Set the model to evaluation mode (important for inference)
model = model.to(device)

# Define transforms (same as used during training)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Same normalization as during training
])

# Directory containing evaluation images (no subfolders)
image_dir = "C:/Users/lusin/Desktop/covid_classifier/data/eval"  # Path to the directory containing evaluation images

# Open the result.txt file for writing predictions
with open('result.txt', 'w') as result_file:
    # Loop through the evaluation set (iterate over all files in the image directory)
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        
        # Skip non-image files
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # Open the image and convert to RGB (in case it's RGBA)
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')  # Convert RGBA or other modes to RGB

        # Apply transformations
        image = data_transforms(image).unsqueeze(0)  # Add batch dimension

        # Move the image tensor to the same device as the model
        image = image.to(device)

        # Get model predictions
        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs, 1)

        # Write results to result.txt
        label = preds.item()  # 0 for non-COVID, 1 for COVID
        result_file.write(f"{image_name} {label}\n")  # Write image name and label (0 or 1)

print("Results saved to result.txt")
