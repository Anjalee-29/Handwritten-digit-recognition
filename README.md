# 🔢 Handwritten Digit Recognition

A neural network built with Keras and TensorFlow that recognizes handwritten digits from the MNIST dataset with high accuracy.

---

## 🛠️ Description

This project trains a fully connected neural network on the MNIST dataset — 70,000 images of handwritten digits (0–9). The model learns to classify each image with high accuracy using:

- A **Dense layer** with 512 neurons and ReLU activation
- A **Dropout layer** (20%) to prevent overfitting
- A **Softmax output layer** for 10-class classification
- **Early stopping** to automatically halt training when validation loss stops improving

The trained model is saved as `mnist_model.keras` after training.

---

## ⚙️ Requirements

- Python 3.8+
- TensorFlow 2.15.0
- Keras 3.0.2

Install all dependencies with:

```bash
pip install -r requirements.txt
```

> **Note:** Training on CPU is supported but may be slow. A GPU with CUDA is recommended for faster training.

---

## 🚀 How to Run

Clone the repository and run the script:

```bash
git clone https://github.com/Anjalee-29/Handwritten-digit-recognition.git
cd Handwritten-digit-recognition
pip install -r requirements.txt
python main.py
```

The MNIST dataset is downloaded automatically on the first run via Keras.

---

## 📊 Expected Results

| Metric | Value |
|---|---|
| Test Accuracy | ~98% |
| Test Loss | ~0.07 |

> Results may vary slightly due to random weight initialization.

---

## 📁 Project Structure

```
├── main.py              # Model definition, training, and evaluation
├── requirements.txt     # Dependencies
├── .gitignore           # Excludes cache, model, and OS files
└── README.md
```

---

## 📝 Notes

- The trained model is saved as `mnist_model.keras` after each run — it is excluded from the repo via `.gitignore` since it can be reproduced by running `main.py`
- Early stopping monitors `val_loss` with a patience of 3 — training typically stops well before 20 epochs
- The MNIST dataset is automatically downloaded and cached by Keras on first run (~11MB)
