import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load model
@st.cache_resource
def load_model():
    model = torch.load('final_model.pth', map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Detect face using OpenCV
def detect_face(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

# Predict emotion
def predict_emotion(face_img):
    face_tensor = transform(face_img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(face_tensor)
        _, predicted = torch.max(outputs, 1)
        return emotion_labels[predicted.item()]

# Streamlit UI
st.title("Facial Emotion Detection")

option = st.radio("Choose input method:", ["📷 Webcam", "📁 Upload Image"])

if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        faces = detect_face(image_np)

        if len(faces) == 0:
            st.warning("No face detected.")
        else:
            for (x, y, w, h) in faces:
                face_crop = image.crop((x, y, x + w, y + h))
                emotion = predict_emotion(face_crop)
                cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image_np, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

            st.image(image_np, caption="Detected Faces", use_column_width=True)

elif option == "📷 Webcam":
    st.warning("Webcam support works locally only.")
    picture = st.camera_input("Take a picture")
    if picture:
        image = Image.open(picture).convert('RGB')
        image_np = np.array(image)
        faces = detect_face(image_np)

        if len(faces) == 0:
            st.warning("No face detected.")
        else:
            for (x, y, w, h) in faces:
                face_crop = image.crop((x, y, x + w, y + h))
                emotion = predict_emotion(face_crop)
                cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image_np, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

            st.image(image_np, caption="Detected Faces", use_column_width=True)