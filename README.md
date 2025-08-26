# Handwritten Alphabet Recognition ✍️

A custom Convolutional Neural Network (CNN) built from scratch with TensorFlow and Keras to classify handwritten English alphabets (A-Z).

## Overview

This project focuses on the fundamental principles of building a CNN for image classification. The model is trained on a large dataset of 28x28 grayscale images of handwritten letters and is designed to accurately identify the corresponding letter of the alphabet for a given image. This serves as a core component for more advanced Optical Character Recognition (OCR) systems.

## 📸 Screenshots

Below is a sample of the model's performance on the unseen test set. The title for each image shows the model's prediction versus the true letter. Green indicates a correct prediction, while red indicates an error.
<img width="1207" height="742" alt="image" src="https://github.com/user-attachments/assets/c76c438a-b012-4bc7-8279-5726a6da12bc" />


## 🛠️ Tech Stack

* **Python 3**
* **TensorFlow & Keras:** For building and training the custom CNN model.
* **Pandas & NumPy:** For loading and preprocessing the CSV data.
* **Matplotlib & Seaborn:** For data visualization and plotting results.
* **Scikit-learn:** For data splitting and one-hot encoding labels.
* **Google Colab:** For training the model with free GPU acceleration.
* **Git & GitHub:** For version control and project hosting.

## Dataset

The model was trained on the **A-Z Handwritten Alphabets in CSV format** dataset from Kaggle. This dataset contains over 370,000 images, with the data structured as a single CSV file where each row represents an image. The first column is the character's label (0-25), and the subsequent 784 columns are the 28x28 pixel values.

You can find the dataset [here](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format).

## 🧠 Model Architecture

Unlike projects that rely on transfer learning, this model is a **custom CNN built from scratch**. The architecture was designed to be lightweight yet effective for this specific task. It consists of:

1.  **Two Convolutional Blocks:** Each block contains a `Conv2D` layer for learning features (like curves and edges) and a `MaxPooling2D` layer to down-sample the feature maps, making the model more efficient.
2.  **A Flatten Layer:** This layer converts the 2D feature maps into a 1D vector.
3.  **Fully Connected Layers:** A `Dense` layer with ReLU activation acts as the classifier, followed by a `Dropout` layer to prevent overfitting.
4.  **Output Layer:** The final `Dense` layer has 26 output neurons (one for each letter of the alphabet) with a `softmax` activation function to produce class probabilities.

## 🚀 Setup and Usage

To run this project yourself, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    ```
2.  **Download the data:**
    * Download the `A_Z Handwritten Data.csv` file from the Kaggle link provided above.
    * Place the `A_Z Handwritten Data.csv` file inside the cloned repository folder.
3.  **Open in Google Colab or Jupyter:**
    * Upload or open the `.ipynb` notebook file.
4.  **Run the notebook:**
    * Execute all cells from top to bottom. The notebook handles all data preprocessing, model training, and evaluation.

## 🎯 Results

The model achieved a high test accuracy of **[Enter Your Accuracy Here, e.g., 99.2%]** on the unseen test data. The results demonstrate the model's excellent ability to generalize and accurately classify handwritten characters, with very few errors on the validation set.
