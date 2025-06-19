import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import cv2
from PIL import Image

# Load the saved MobileNet model
MODEL_PATH = os.path.join(os.getcwd(), 'interrupted_model.keras')  # MobileNet weights on 22 epochs
model = load_model(MODEL_PATH)

# Define emotion labels
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to preprocess the input image
def preprocess_image(image):
    # Resize image to 224x224 (required by MobileNet)
    img = image.resize((224, 224))
    # Convert to numpy array and scale pixel values
    img_array = img_to_array(img) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit app
st.title("Facial Emotion Recognition App")
st.write("Upload an image, and the model will predict the facial emotion!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert BGR to RGB for displaying in Streamlit
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    # Load a pre-trained Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw green rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert back to PIL image for displaying in Streamlit
    image_with_boxes = Image.fromarray(image_rgb)

    # Display the image with green boxes
    st.image(image_with_boxes, caption='Uploaded Image with Detected Faces', use_column_width=True)
    st.write("Classifying...")

    if len(faces) > 0:
        # Preprocess the first detected face for emotion recognition
        x, y, w, h = faces[0]
        face_image = image_rgb[y:y + h, x:x + w]
        face_pil = Image.fromarray(face_image)
        preprocessed_image = preprocess_image(face_pil)

        # Predict emotion
        predictions = model.predict(preprocessed_image)
        emotion_index = np.argmax(predictions)
        emotion_label = EMOTION_LABELS[emotion_index]

        # Display the result
        st.write(f"Facial Emotion: **{emotion_label}**")
    else:
        st.write("No face detected. Please try a different image.")
