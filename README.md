# 🔢 Digits Classifier

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge\&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?style=for-the-badge\&logo=tensorflow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow?style=for-the-badge\&logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

### Handwritten Digit Recognition using Machine Learning & Deep Learning

A professional machine learning project that classifies handwritten digits (0–9) using computer vision and neural networks trained on the MNIST dataset.

</div>

---

# 📌 Overview

This project demonstrates the implementation of a handwritten digit classification system capable of recognizing numerical digits from images using deep learning techniques.

The model is trained on the famous **MNIST dataset**, which contains thousands of grayscale images of handwritten digits. The system learns visual patterns and predicts the correct digit with high accuracy.

This repository highlights practical skills in:

* Machine Learning
* Deep Learning
* Computer Vision
* Neural Networks
* Data Preprocessing
* Model Evaluation
* Python Development

---

# 🚀 Key Features

✅ Handwritten digit recognition (0–9)

✅ Neural network / CNN-based architecture

✅ Trained on the MNIST dataset

✅ Data preprocessing and normalization pipeline

✅ Model training and evaluation workflow

✅ Prediction support for custom images

✅ Easy-to-run Python implementation


---

# 🧠 Problem Statement

Handwritten digit recognition is a foundational computer vision task widely used in:

* Banking systems (cheque processing)
* Postal code recognition
* OCR systems
* Educational tools
* Form digitization
* AI-powered automation systems

The objective of this project is to build an intelligent model capable of automatically recognizing handwritten digits with high accuracy.

---

# 🛠️ Tech Stack

## Languages & Frameworks

* Python
* TensorFlow / Keras 
* NumPy
* Matplotlib
* OpenCV
* Scikit-learn

## Machine Learning Concepts

* Convolutional Neural Networks (CNN)
* Image Classification
* Supervised Learning
* Model Evaluation
* Feature Extraction

---

# 📂 Suggested Repository Structure

To improve readability and recruiter experience, consider organizing the repository like this:

```bash
Digits-Classifier/
│
├── data/                   # Dataset storage
├── models/                 # Saved trained models
├── src/
│   ├── train.py            # Model training
│   └── app.py              # Run project 
│
├── assets/
│   ├── demo.png            # Prediction examples
│   ├── confusion_matrix.png
│   └── architecture.png
│
├── tests/                  # Unit tests
├── requirements.txt
├── README.md
└── LICENSE
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/rohaanahmadtech/digits_classifier.git
cd digits_classifier
```

## 2️⃣ Create Virtual Environment *(Recommended)*

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Example dependencies:

```txt
numpy
matplotlib
tensorflow
opencv-python
scikit-learn
pandas
```

---

# ▶️ Running the Project

## Train the Model

```bash
python src/train.py
```

## Run Prediction

```bash
python src/predict.py
```

## Launch Jupyter Notebook *(Optional)*

```bash
jupyter notebook
```

---

# 📸 Usage Example

## Example Input

A grayscale image containing a handwritten digit.

```python
prediction = model.predict(image)
print("Predicted Digit:", prediction)
```

## Example Output

```bash
Predicted Digit: 7
Confidence: 99.2%
```

---

# 🧪 Dataset Information

## MNIST Dataset

The project uses the **MNIST handwritten digits dataset**, which contains:

| Dataset Split | Images          |
| ------------- | --------------- |
| Training Set  | 60,000          |
| Testing Set   | 10,000          |
| Image Size    | 28 × 28         |
| Classes       | 10 Digits (0–9) |

### Dataset Characteristics

* Grayscale handwritten digit images
* Standard benchmark dataset for computer vision
* Widely used in machine learning research

---

# 🔍 Data Preprocessing

The preprocessing pipeline includes:

* Image normalization
* Pixel scaling (0–255 → 0–1)
* Reshaping input dimensions
* One-hot encoding labels
* Noise reduction *(if applicable)*

Example preprocessing:

```python
X = X / 255.0
X = X.reshape(-1, 28, 28, 1)
```

---

# 🧠 Model Architecture

The project uses a Convolutional Neural Network (CNN) architecture for image classification.

## Example Architecture

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

---

# 📊 Training Details

| Parameter     | Value                    |
| ------------- | ------------------------ |
| Optimizer     | Adam                     |
| Loss Function | Categorical Crossentropy |
| Batch Size    | 32                       |
| Epochs        | 10–20                    |
| Framework     | TensorFlow / Keras       |

---

# 📈 Model Performance

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

## Example Results

| Metric              | Score |
| ------------------- | ----- |
| Training Accuracy   | 99%   |
| Validation Accuracy | 98%   |
| Test Accuracy       | 98%+  |

### Performance Highlights

* High digit recognition accuracy
* Efficient inference time
* Robust performance on handwritten samples
* Demonstrates practical CNN implementation skills

---

# 📷  Visual Assets

* Model architecture diagram
* Confusion matrix image
* Sample predictions
* Training accuracy/loss graphs
* Demo GIF or screenshots

Example:

```markdown
![Model Architecture](assets/architecture.png)
![Confusion Matrix](assets/confusion_matrix.png)
```

---

# 📌 Skills Demonstrated

This project demonstrates practical experience in:

* Deep Learning
* CNN Architecture Design
* Computer Vision
* Model Optimization
* Data Processing Pipelines
* Python Development
* AI Model Evaluation
* Research & Experimentation
* Git & GitHub Collaboration

---


# 🛣️ Roadmap / Future Improvements

Planned enhancements:

* Deploy model using Streamlit or Flask
* Add real-time digit drawing canvas
* Improve model accuracy with advanced CNNs
* Add Docker support
* Add CI/CD workflow using GitHub Actions
* Convert model to ONNX / TensorFlow Lite
* Mobile deployment support
* Add automated testing

---

# 📜 License

This project is licensed under the MIT License.

```txt
MIT License © 2026 Rohaan Ahmad
```

---

# 👨‍💻 Author

## Rohaan Ahmad

BSCS Student | AI & ML Enthusiast | Web Developer

Passionate about building intelligent systems using machine learning and modern web technologies.

### Connect With Me

* GitHub: [https://github.com/rohaanahmadtech](https://github.com/rohaanahmadtech)
* LinkedIn: [www.linkedin.com/in/rohaanahmad07](https://linkedin.com/in/rohaanahmad07/)
* Email: [rohaanahmad.tech@gmail.com](mailto:rohaanahmad.tech@gmail.com)

---

# ⭐ Support the Project

If you found this project useful:

⭐ Star the repository

🍴 Fork the project

📢 Share it with others

---


<div align="center">

### Built with ❤️ using Machine Learning & Deep Learning

</div>
