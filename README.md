![Star Badge](https://img.shields.io/static/v1?label=%F0%9F%8C%9F&message=If%20Useful&style=flat&color=BC4E99)
![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)

# 🔢 MNIST Digit Recognizer

A high-accuracy digit recognizer built with Keras and TensorFlow. Train a CNN on the MNIST dataset, then draw a digit or upload an image to see the model predict it in real time!

---

## 🛠️ Description

This project has two parts:

- **`main.py`** — Trains a CNN on the MNIST dataset (60,000 handwritten digit images) and saves the model
- **`predict.py`** — Opens a GUI where you can either draw a digit or upload an image, and the model predicts it with a confidence score

The model architecture:
- **3 Convolutional blocks** with increasing filters (32 → 64 → 128) and ReLU activation
- **BatchNormalization** after every block for stable, faster training
- **MaxPooling** and **Dropout** layers to reduce overfitting
- **Fully connected head** with 512 → 256 → 10 neurons
- **Softmax output** for 10-class classification
- **Data Augmentation** — random rotation, zoom, and translation to improve robustness
- **Early stopping** with best weight restoration
- **ReduceLROnPlateau** — automatically reduces learning rate when training stalls

---

## ⚙️ Requirements

- Python 3.8+
- TensorFlow 2.20+
- Keras 3.0+
- Pillow
- tkinter (built into Python on Windows/macOS)

Install all dependencies:

```bash
pip install -r requirements.txt
```

> **Linux users** may need to install tkinter separately:
> ```bash
> sudo apt-get install python3-tk
> ```

---

## 🚀 How to Run

Clone the repository:

```bash
git clone https://github.com/Anjalee-29/Handwritten-digit-recognition.git
cd Handwritten-digit-recognition
pip install -r requirements.txt
```

### Step 1 — Train the model:

```bash
python main.py
```

This downloads the MNIST dataset automatically, trains the CNN, and saves it as `mnist_model.keras`.

### Step 2 — Recognize a digit:

```bash
python predict.py
```

Two modes are available:

**✏️ Draw mode (default):**
- Draw a digit on the black canvas using your mouse
- Click **Predict** to see the result and confidence score
- Click **Clear** to draw again

**🖼️ Upload mode:**
- Switch to the **Upload Image** tab
- Click **Browse Image** to select a `.png`, `.jpg`, `.bmp`, or `.tiff` file
- A preview of the image is shown
- Click **Predict** to see the result and confidence score

---

## 📊 Expected Results

| Metric | Old Model (Dense) | New Model (CNN) |
|---|---|---|
| Test Accuracy | ~98% | ~99.3–99.5% |
| Test Loss | ~0.07 | ~0.02 |

---

## 📁 Project Structure

```
├── main.py              # CNN model training and evaluation
├── predict.py           # GUI for drawing or uploading digits
├── requirements.txt     # Dependencies
├── .gitignore           # Excludes cache and model files
└── README.md
```

