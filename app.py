from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import gdown
import os

app = Flask(__name__)
CORS(app)

# Google Drive URL for the model
gdrive_url = 'https://drive.google.com/uc?id=18haQGGA39LUcoPb5GWflIY7-jWcgwxlB'
model_path = 'best_densenet121_model.pth'

# Download the model file if it doesn't exist
if not os.path.exists(model_path):
    gdown.download(gdrive_url, model_path, quiet=False)

# Define the model architecture (DenseNet121 in this case)
model = models.densenet121(weights=None, num_classes=4)  # Adjust num_classes as per your model

# Load the state dictionary
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Define the image preprocessing function
def preprocess_image(image_bytes):
    # Open image using PIL
    image = Image.open(image_bytes)

    # Convert grayscale to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Apply transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    return input_batch

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    input_tensor = preprocess_image(file)

    # Model inference
    with torch.no_grad():
        output = model(input_tensor)
    prediction = output.argmax(dim=1).item()

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
