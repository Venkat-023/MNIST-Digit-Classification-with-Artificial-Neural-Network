# 🖐️ MNIST Handwritten Digit Classification using Artificial Neural Network (ANN)

This project implements a fully connected Artificial Neural Network (ANN) to classify handwritten digits (0-9) from the MNIST dataset. The model achieves a high accuracy of **97.84%** on the test set by leveraging dense layers with ReLU activation and dropout regularization to prevent overfitting.

---

## 📌 Problem Statement

The objective is to accurately classify grayscale images of handwritten digits (28x28 pixels) into one of 10 classes (digits 0 through 9). This is a classic image classification problem widely used as a benchmark for machine learning algorithms.

---

## 🧠 Model Architecture

The neural network is designed as follows:

- **Input Layer:** Flatten 28x28 pixel images into a vector of 784 features
- **Hidden Layer 1:** Dense layer with 256 neurons, ReLU activation
- **Dropout Layer:** Dropout with rate 0.5 to reduce overfitting
- **Hidden Layer 2:** Dense layer with 128 neurons, ReLU activation
- **Dropout Layer:** Dropout with rate 0.3
- **Output Layer:** Dense layer with 10 neurons and softmax activation for multiclass classification

```python
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))
📈 Training and Evaluation
Accuracy: 97.84% on test data

Loss and Accuracy Graphs: Plotted training vs validation loss and accuracy to monitor model performance and detect overfitting

Used categorical crossentropy loss and Adam optimizer

 Libraries and Tools
TensorFlow / Keras (for building and training the ANN)

NumPy (for numerical operations)

Matplotlib (for plotting training and validation curves)

scikit-learn (optional, for metrics and data handling)

▶️ How to Run
Clone the repository:
git clone https://github.com/<your-username>/mnist-ann.git
cd mnist-ann
Install dependencies:

pip install -r requirements.txt
Run the notebook or script:

jupyter notebook MNIST_ANN.ipynb
