# Face Recognition with FER2013 using CNN

This project is a Convolutional Neural Network (CNN) implementation for face recognition and emotion detection using the FER2013 dataset. The model is trained to classify facial expressions into seven distinct emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to build and train a CNN model for recognizing facial expressions in images using the FER2013 dataset. The model architecture includes multiple convolutional layers followed by max-pooling and dropout layers to prevent overfitting.

## Dataset

The FER2013 dataset consists of 48x48 grayscale images of faces, each labeled with one of seven emotions:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Model Architecture

The model consists of the following layers:
1. **Conv2D**: 32 filters, 3x3 kernel, ReLU activation
2. **MaxPooling2D**: 2x2 pool size
3. **Dropout**: 0.25 rate
4. **Conv2D**: 64 filters, 3x3 kernel, ReLU activation
5. **MaxPooling2D**: 2x2 pool size
6. **Dropout**: 0.25 rate
7. **Conv2D**: 128 filters, 3x3 kernel, ReLU activation
8. **MaxPooling2D**: 2x2 pool size
9. **Dropout**: 0.25 rate
10. **Flatten** layer
11. **Dense**: 512 units, ReLU activation
12. **Dropout**: 0.5 rate
13. **Dense**: 7 units, Softmax activation

## Installation

To install the required dependencies, run:
    ```bash
    pip install -r requirements.txt
    ```

Usage
To train the model, use:

history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])

To evaluate the model and generate a confusion matrix:

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Test set predictions
y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)
y_test_class = np.argmax(y_test, axis=1)

# Generate the classification report
report = classification_report(y_test_class, y_pred_class, target_names=emotion_labels)
print(report)

# Generate and plot the confusion matrix
cm = confusion_matrix(y_test_class, y_pred_class)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

Evaluation
The model is evaluated using the accuracy and loss metrics, as well as precision, recall, and F1-score to understand the performance across different classes.

Results
The model's performance metrics and the confusion matrix will be displayed, illustrating the accuracy and the classification report across the different emotion categories.

Contributing
Contributions are welcome! Please open an issue or submit a pull request.

License
This project is licensed under the MIT License.
