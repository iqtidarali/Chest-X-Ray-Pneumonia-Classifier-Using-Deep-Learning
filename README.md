
## Chest X-Ray Pneumonia Classifier

A beginner deep learning project that classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using a pretrained ResNet18 model.

---

## 📌 Project Overview

| | |
|---|---|
| **Type** | Binary Image Classification |
| **Model** | ResNet18 (pretrained on ImageNet) |
| **Dataset** | [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| **Classes** | NORMAL · PNEUMONIA |
| **Test Accuracy** | 90% |

---

## 📁 Dataset Structure

After downloading and unzipping, your folder should look like this:

```
chest_xray/
  train/
    NORMAL/        → 1,341 images
    PNEUMONIA/     → 3,875 images
  val/
    NORMAL/        → 8 images
    PNEUMONIA/     → 8 images
  test/
    NORMAL/        → 234 images
    PNEUMONIA/     → 390 images
```

---

## ▶️ How to Run

**Option 1 — Google Colab (recommended, free GPU)**
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload the notebook
3. Set runtime to GPU: `Runtime → Change runtime type → GPU`
4. Run all cells top to bottom

**Option 2 — Local Jupyter**
```bash
pip install jupyter
jupyter notebook chest_xray_classifier.ipynb
```

---

## 📓 Notebook Walkthrough

### Cell 1 — Install Libraries
Installs all required Python packages: `torch`, `torchvision`, `matplotlib`, `pillow`, `tqdm`, `scikit-learn`, and `seaborn`. Run this once before anything else.

---

### Cell 2 — Download Dataset
Downloads the chest X-ray dataset from Kaggle using the Kaggle API. After downloading, it checks that all folders exist and prints the image count for each split (train / val / test). You need a free Kaggle account and API key (`kaggle.json`) for this step.

---

### Cell 3 — Explore & Visualise Sample Images
Displays a grid of random X-ray images from the training set — 4 NORMAL images on top and 4 PNEUMONIA images below. This helps you visually understand what the two classes look like before training. PNEUMONIA lungs typically appear cloudier or hazier than normal ones.

---

### Cell 4 — Preprocess & Load Data
Prepares the images for the model:
- **Resize** all images to 224×224 pixels (required by ResNet18)
- **Augmentation** on training images (random flip, rotation, brightness change) — this artificially expands the dataset and prevents overfitting
- **Normalize** pixel values using ImageNet statistics so the pretrained model works correctly
- Wraps everything in `DataLoader` for efficient batch loading during training

---

### Cell 5 — Build Model (Pretrained ResNet18)
Loads ResNet18, a powerful image classifier already trained on 1.2 million images (ImageNet). We:
- **Freeze** all its layers — so we don't change what it already learned
- **Replace only the final layer** to output 2 classes (NORMAL / PNEUMONIA) instead of 1000

This technique is called **transfer learning** — reusing a strong model for a new task. Only 1,026 parameters are trained instead of 11 million.

---

### Cell 6 — Set Up Loss & Optimizer
Defines how the model learns:
- **CrossEntropyLoss** — measures how wrong the predictions are
- **Adam optimizer** — adjusts the final layer weights to reduce the loss
- **Learning rate scheduler** — automatically lowers the learning rate if validation loss stops improving, helping the model fine-tune more carefully

---

### Cell 7 — Train the Model
Runs the training loop for 5 epochs (full passes through the data). Each epoch:
1. Feeds batches of images through the model
2. Calculates the loss (how wrong the prediction was)
3. Updates the model weights to do better next time
4. Evaluates performance on the validation set
5. Saves the best model automatically

**Results achieved:**

| Epoch | Train Acc | Val Acc |
|---|---|---|
| 1 | 85.05% | 81.25% |
| 2 | 90.26% | 81.25% |
| 3 | 91.81% | 81.25% |
| 4 | 92.06% | 87.50% ✅ |
| 5 | 92.33% | 87.50% |

---

### Cell 8 — Plot Training Curves
Draws two charts side by side:
- **Loss over epochs** — should decrease over time
- **Accuracy over epochs** — should increase over time

These curves help you spot problems like overfitting (training accuracy much higher than validation). Saved as `training_curves.png`.

---

### Cell 9 — Evaluate on Test Set + Confusion Matrix
Loads the best saved model and runs it on the test set (images the model has never seen). Reports:
- **Precision, Recall, F1-score** per class
- **Overall accuracy: 90%**
- **Confusion matrix** — shows exactly how many images were correctly and incorrectly classified

**Test Results:**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| NORMAL | 0.93 | 0.80 | 0.86 |
| PNEUMONIA | 0.89 | 0.96 | 0.92 |
| **Overall** | | | **0.90** |

---

### Cell 10 — Predict on a Single Image
Takes one X-ray image and outputs:
- The predicted class (NORMAL or PNEUMONIA)
- The confidence percentage
- Probability for each class

Example output:
```
Result     : PNEUMONIA
Confidence : 98.2%
  NORMAL: 1.8%
  PNEUMONIA: 98.2%
```

---

## 📊 Results Summary

| Metric | Value |
|---|---|
| Test Accuracy | 90% |
| PNEUMONIA Recall | 96% |
| NORMAL Precision | 93% |
| Best Val Accuracy | 87.5% |

The model detects pneumonia with **96% recall** — meaning it catches nearly all pneumonia cases, which is important in a medical setting where missing a diagnosis has serious consequences.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| PyTorch | Model building and training |
| TorchVision | Pretrained ResNet18 + image transforms |
| Matplotlib | Plots and image visualization |
| Scikit-learn | Classification report and confusion matrix |
| Seaborn | Confusion matrix heatmap |
| tqdm | Progress bars during training |

---

## 🚀 Try This

- Add **Grad-CAM** to visualize which region of the X-ray the model focuses on
- Try **EfficientNet** for higher accuracy
- Train on the full **NIH Chest X-ray Dataset** (100k+ images)
- Build a **Streamlit web app** for uploading and classifying your own X-rays
- Move to **Stage 2**: Brain Tumor MRI Classification


## 📄 License

This project is for educational purposes. The dataset belongs to [Kaggle / Paul Mooney](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
