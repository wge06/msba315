import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

# ---------------------
# Define your model class or structure here (example CNN)
# ---------------------
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),  # Adjust if input size is different
            nn.ReLU(),
            nn.Linear(128, 5)  # 5 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------------------
# Load the model safely
# ---------------------
@st.cache_resource
def load_model():
    model = EmotionCNN()
    model.load_state_dict(torch.load("final_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
classes = ['angry', 'happy', 'neutral', 'sad', 'surprised']

# ---------------------
# Image preprocessing
# ---------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_emotion(image: Image.Image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return classes[predicted.item()]

# ---------------------
# Streamlit UI
# ---------------------
st.title("Facial Emotion Detection")
st.write("Detect human emotions from webcam or image upload")

option = st.radio("Choose input method:", ("Upload Image", "Use Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        label = predict_emotion(image)
        st.success(f"Predicted Emotion: **{label}**")

elif option == "Use Webcam":
    st.warning("Streamlit Cloud does not support webcam access directly.")
    st.info("Please run this locally using: `streamlit run app.py`")
