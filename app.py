import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import cv2
import numpy as np
import tempfile
import os

# Define the emotion labels
EMOTION_LABELS = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprised']
NUM_CLASSES = 5
# Load the model
@st.cache_resource
def load_model():
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    )
    model.load_state_dict(torch.load("final_model.pth"))
    model.eval()
    return model

model = load_model()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# Face detection using OpenCV Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces_and_predict(image):
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    annotated_img = img_array.copy()
    for (x, y, w, h) in faces:
        face = img_array[y:y+h, x:x+w]
        face_pil = Image.fromarray(face)
        input_tensor = transform(face_pil).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            label = EMOTION_LABELS[predicted.item()]
        # Annotate
        cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(annotated_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    return annotated_img

# Streamlit UI
st.title("MSBA 315 Project - Facial Emotion Detection")

option = st.radio("Choose input type:", ("Use Webcam", "Upload an Image"))

if option == "Upload an Image":
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        result_img = detect_faces_and_predict(image)
        st.image(result_img, caption='Detected Emotions', use_container_width=True)

elif option == "Use Webcam":
    picture = st.camera_input("Take a picture")
    if picture:
        image = Image.open(picture)
        result_img = detect_faces_and_predict(image)
        st.image(result_img, caption='Detected Emotions', use_container_width=True)
