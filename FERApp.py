import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

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

# Streamlit app layout
st.title("Facial Emotion Recognition")
st.caption("Upload an image to predict the facial emotion.")  # Smaller description text

# Organize the layout in columns to minimize scrolling
with st.container():
    col1, col2 = st.columns([1, 1])  # Two equally-sized columns

    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png'], 
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)

    with col2:
        if uploaded_file is not None:
            st.write("Classifying...")
            # Load the uploaded image
            image = load_img(uploaded_file)

            # Preprocess the image
            preprocessed_image = preprocess_image(image)

            # Predict emotion
            predictions = model.predict(preprocessed_image)
            emotion_index = np.argmax(predictions)
            emotion_label = EMOTION_LABELS[emotion_index]

            # Display the result
            st.markdown(f"<h3 style='text-align: center;'>Facial Emotion:</h3>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center; color: blue;'>{emotion_label}</h2>", unsafe_allow_html=True)
        else:
            st.info("Upload an image to get started.")

# Styling adjustments to reduce margin/padding
st.markdown(
    """
    <style>
    .block-container {
        padding: 1rem;
    }
    img {
        margin: 0 auto;
        display: block;
    }
    </style>
    """,
    unsafe_allow_html=True
)
