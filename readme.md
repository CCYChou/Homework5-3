# CIFAR-10 Classification using Pretrained VGG16 and VGG19

## Project Description
This project aims to classify images from the CIFAR-10 dataset into 10 categories using pretrained VGG16 and VGG19 models. Transfer learning is employed to adapt these models for the CIFAR-10 dataset. The code includes data preprocessing, model training, evaluation, and visualizations.

---

## Steps Overview

### 1. Business Understanding
Objective: Classify CIFAR-10 images into 10 categories: 
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

### 2. Data Understanding
The CIFAR-10 dataset contains:
- **60,000 images** of size 32x32 pixels.
- 10 classes with 6,000 images per class.

### 3. Data Preparation
- **Normalization:** Pixel values scaled to the range [0, 1].
- **One-Hot Encoding:** Labels converted to categorical format.
- **Data Augmentation:** Rotation, width/height shifts, and horizontal flips to increase data diversity.

### 4. Modeling
- **Pretrained Models Used:** VGG16 and VGG19.
- **Modifications:**
  - Flattening the output of the convolutional base.
  - Adding a dense layer with 256 neurons and ReLU activation.
  - Adding a dropout layer with a rate of 0.5.
  - Output layer with 10 neurons and softmax activation.

### 5. Evaluation
- **Callbacks:** Early stopping and model checkpointing.
- **Metrics:** Accuracy and loss for training and validation datasets.
- **Classification Report:** Precision, recall, and F1-score per class.

### 6. Deployment
- Model weights saved as `vgg16_cifar10.h5` and `vgg19_cifar10.h5`.
- Evaluation results and metrics provided in classification reports.

### 7. Visualizations
- Plots for training and validation accuracy and loss over epochs.

---

## Prerequisites

### Python Libraries
- `tensorflow`
- `numpy`
- `matplotlib`
- `sklearn`

### Environment
Ensure that TensorFlow is correctly set up with GPU support for faster training. Follow [TensorFlow's official GPU setup guide](https://www.tensorflow.org/install/gpu) if needed.

---

## How to Run the Code

1. Clone the repository or download the script.
2. Install the required libraries:
    ```bash
    pip install tensorflow numpy matplotlib scikit-learn
    ```
3. Run the script:
    ```bash
    python cifar10_vgg_classification.py
    ```
4. The script will:
    - Preprocess the CIFAR-10 dataset.
    - Train the VGG16 and VGG19 models.
    - Save the trained models.
    - Display evaluation metrics and visualizations.

---

## Output Files
- **Trained Models:**
  - `vgg16_cifar10.h5`
  - `vgg19_cifar10.h5`

- **Visualizations:** Training and validation accuracy/loss plots.

---

## Results
- Evaluation metrics (accuracy, precision, recall, F1-score) for both models are displayed in the console.
- Saved models can be reused for predictions on new CIFAR-10-like datasets.

---

## References
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [TensorFlow Documentation](https://www.tensorflow.org/)

---

## Contact
For questions or feedback, please reach out to [Your Name/Email].
