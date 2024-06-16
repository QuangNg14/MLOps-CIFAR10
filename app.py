import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from PIL import Image
import io

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


# Model serving with Flask
app = Flask(__name__)
model = SimpleCNN().to(device)


def transform_image(image_bytes):
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


@app.route("/", methods=["GET"])
def hello():
    return "Hello, World!"


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        img_bytes = file.read()
        tensor = transform_image(img_bytes)
        output = model(tensor)
        _, predicted = torch.max(output.data, 1)
        return jsonify({"prediction": predicted.item()})


if __name__ == "__main__":
    model_path = "models/simple_cnn.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        app.run(debug=True)
    else:
        print(
            f"Model file not found at {model_path}. Please ensure the model is trained and saved correctly."
        )
