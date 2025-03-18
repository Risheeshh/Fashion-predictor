# Fashion MNIST Image Classification

## Project Overview
This project classifies images of clothing items from the **Fashion MNIST** dataset using deep learning. It provides two different approaches:
- **Custom CNN Model**: A convolutional neural network (CNN) trained from scratch.
- **MobileNetV2**: A pre-trained deep learning model adapted for Fashion MNIST classification.

## Project Components
The project consists of three key scripts:

### 1. `train.py` (Training Script - Custom Model)
**Purpose**: Trains a **custom CNN model** and saves it as `model.keras`.

**Key Features**:
- Loads the **Fashion MNIST** dataset (28×28 grayscale images).
- Preprocesses and normalizes images.
- Defines a **custom CNN architecture** for classification.
- Trains the model on the dataset.
- Saves the trained model as `model.keras`.

### 2. `new.py` (Training & Prediction - MobileNetV2)
**Purpose**: Uses **MobileNetV2**, a pre-trained model, to classify **Fashion MNIST images** and external images.

**Key Features**:
- Loads the **Fashion MNIST** dataset.
- Converts grayscale images to **RGB (3 channels)** and resizes them to **224×224**.
- Uses **MobileNetV2** with additional dense layers.
- Trains and saves the model as `model_mobilenetv2.keras`.
- Includes a **prediction function**:
  - Reads and processes an external image.
  - Resizes and normalizes it.
  - Uses the trained model to classify the image.

### 3. `predict.py` (Prediction Script for `model.keras`)
**Purpose**: Loads `model.keras` and classifies a new image provided by the user.

**How It Works**:
- Loads `model.keras` (trained using `train.py`).
- Reads an input image in grayscale and resizes it to **28×28**.
- **Inverts pixel values** (to improve contrast with training images).
- Normalizes the image and feeds it into the model.
- Predicts and outputs the clothing category.

---

## Project Files Summary

| File | Purpose |
|------|---------|
| `train.py` | Trains a custom CNN model and saves it as `model.keras` |
| `new.py` | Uses MobileNetV2 to train a separate model (`model_mobilenetv2.keras`) and classify images |
| `predict.py` | Loads `model.keras` and predicts clothing items from an input image |
| `model.keras` | Pre-trained custom CNN model for predictions |
| `model_mobilenetv2.keras` | Pre-trained MobileNetV2 model for predictions |

---

## How to Run This Project

### **1. Train a Model**
You can train either model:
#### **Option 1: Train Custom CNN**
```bash
python train.py
```
- Generates `model.keras`.

#### **Option 2: Train MobileNetV2**
```bash
python new.py
```
- Generates `model_mobilenetv2.keras`.

### **2. Make Predictions**
#### **Using `model.keras` (28×28 images)**
```bash
python predict.py
```
- Prompts for an image and predicts the clothing category.

#### **Using `model_mobilenetv2.keras` (External Images, 224×224)**
```bash
python new.py
```
- Prompts for an image and predicts the clothing category.

---

## Possible Enhancements
- Improve the **CNN architecture** in `train.py` for better accuracy.
- Fine-tune **MobileNetV2** for better classification on Fashion MNIST.
- Add a **GUI** for easier image input and visualization.
- Expand the dataset to include **real-world clothing images** for better generalization.

---

## Conclusion
This project explores two deep learning approaches for **Fashion MNIST image classification**:
1. A **custom CNN** for training from scratch.
2. A **pre-trained MobileNetV2** for transfer learning.

It provides flexibility for training and prediction, allowing classification of both dataset images and external images.

Happy coding!

