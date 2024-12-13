import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Business Understanding
# Objective: Classify CIFAR-10 images into 10 categories using pretrained VGG16 and VGG19 models.

# Step 2: Data Understanding
# CIFAR-10 dataset contains 60,000 32x32 color images in 10 classes, with 6,000 images per class.

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Class names
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Step 3: Data Preparation
# Normalize the data to the range [0, 1] and one-hot encode the labels.
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encoding
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# Step 4: Modeling
# Function to create a model based on a pretrained VGG architecture
def create_vgg_model(base_model, input_shape, num_classes):
    base_model = base_model(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model

input_shape = (32, 32, 3)

# Create VGG16 model
vgg16_model = create_vgg_model(VGG16, input_shape, num_classes)
vgg16_model.compile(optimizer=Adam(learning_rate=0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

# Create VGG19 model
vgg19_model = create_vgg_model(VGG19, input_shape, num_classes)
vgg19_model.compile(optimizer=Adam(learning_rate=0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

# Step 5: Evaluation
# Define callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True)
]

# Train VGG16
history_vgg16 = vgg16_model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    validation_data=(x_test, y_test),
    epochs=50,
    callbacks=callbacks
)

# Train VGG19
history_vgg19 = vgg19_model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    validation_data=(x_test, y_test),
    epochs=50,
    callbacks=callbacks
)

# Step 6: Deployment
# Evaluate the models on the test set
print("Evaluating VGG16...")
vgg16_eval = vgg16_model.evaluate(x_test, y_test, verbose=0)
print(f"VGG16 Test Accuracy: {vgg16_eval[1]:.4f}")

print("Evaluating VGG19...")
vgg19_eval = vgg19_model.evaluate(x_test, y_test, verbose=0)
print(f"VGG19 Test Accuracy: {vgg19_eval[1]:.4f}")

# Classification report for VGG16
vgg16_predictions = np.argmax(vgg16_model.predict(x_test), axis=1)
y_test_labels = np.argmax(y_test, axis=1)
print("Classification Report for VGG16:")
print(classification_report(y_test_labels, vgg16_predictions, target_names=class_names))

# Classification report for VGG19
vgg19_predictions = np.argmax(vgg19_model.predict(x_test), axis=1)
print("Classification Report for VGG19:")
print(classification_report(y_test_labels, vgg19_predictions, target_names=class_names))

# Save models
vgg16_model.save("vgg16_cifar10.h5")
vgg19_model.save("vgg19_cifar10.h5")

# Step 7: Visualizations
import matplotlib.pyplot as plt

# Plot training and validation accuracy
for history, label in zip([history_vgg16, history_vgg19], ["VGG16", "VGG19"]):
    plt.plot(history.history['accuracy'], label=f"{label} Train")
    plt.plot(history.history['val_accuracy'], label=f"{label} Validation")
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(loc="lower right")
plt.show()

# Plot training and validation loss
for history, label in zip([history_vgg16, history_vgg19], ["VGG16", "VGG19"]):
    plt.plot(history.history['loss'], label=f"{label} Train")
    plt.plot(history.history['val_loss'], label=f"{label} Validation")
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(loc="upper right")
plt.show()
