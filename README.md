# 😄 Facial Emotion Recognition (FER) System

A deep learning project for classifying facial expressions into 7 emotion categories using Convolutional Neural Networks (CNN) and Transfer Learning (MobileNet). Includes a real-time emotion prediction interface built with Streamlit.

## 📁 Dataset Description

- **Image Size**: 48x48 grayscale images
- **Classes**:
  - Angry (3,995)
  - Disgust (436)
  - Fear (4,097)
  - Happy (7,215)
  - Sad (4,830)
  - Surprise (3,171)
  - Neutral (4,965)
- **Total Samples**: 28,709 training | 3,589 test

## 🧠 Models and Architecture

### CNN-Based Model:
- 3 Conv blocks (32, 64, 128 filters)
- BatchNorm, MaxPooling, Dropout
- Dense(256) + Softmax output

### Transfer Learning with MobileNet:
- Pre-trained MobileNet used as a frozen feature extractor
- Custom classifier layers added
- Finetuning with small learning rate

## ⚙️ Implementation Details

- **Tools**: TensorFlow, NumPy, Matplotlib, Seaborn, Scikit-learn, Visual Studio
- **Preprocessing**: 
  - Rescaling [0, 1]
  - Data augmentation (flip, rotate, shift, shear, zoom)
- **Training**:
  - Optimizer: Adam
  - Loss: Categorical Crossentropy
  - Batch size: 32
  - Epochs: 22
  - Early stopping (patience: 5)


## 📊 Evaluation

| Metric        | Value         |
|---------------|---------------|
| Training Acc  | 42.25%        |
| Validation Acc| 50.35%        |

- F1-score used due to class imbalance
- Confusion matrix for class-wise accuracy

## 🖥️ User Interface (Streamlit)

Features:
- Upload grayscale facial image
- Preprocessing and classification
- Live prediction with emotion label
- Display of uploaded image with prediction


## 🚧 Limitations & Future Work

### ❌ Limitations:
- Computational limits (trained for 10–22 epochs)
- Class imbalance (e.g., Disgust class underrepresented)
- Long training times due to limited hardware

## 📂 Folder Structure Suggestion

facial-emotion-recognition/
│
├── data/
│ └── fer2013.csv
│
├── models/
│ ├── cnn_model.h5
│ └── mobilenet_model.h5
│
├── streamlit_app.py
├── train_cnn.py
├── train_mobilenet.py
├── README.md
└── requirements.txt

## DEMO

![Model Accuracy](model_accuracy.png)
