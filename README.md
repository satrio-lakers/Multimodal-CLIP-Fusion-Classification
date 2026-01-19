# Multimodal Classification Using Transfer Learning (CLIP) & Logistic Regression

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![HuggingFace](https://img.shields.io/badge/Transformers-CLIP-yellow) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Logistic%20Regression-green)

## 📌 Project Description
This project is a **Multimodal Machine Learning** solution that combines **Text** and **Image** modalities to classify data (such as memes or social media content) into binary categories (e.g., *Offensive* vs. *Non-Offensive*).

The model is built to understand the combined context between visuals and narrative sentences using a **Feature Extraction** approach.

## 🎯 Primary Goal
To build a model capable of predicting classification labels with high accuracy by understanding the dual semantic context (image & text). The core strategy involves using **Deep Learning** (CLIP) to convert raw data into numerical vectors, which are then classified by an efficient statistical model.

## 🏗️ Architecture & Technology

* **Core Model (Backbone):** `openai/clip-vit-base-patch32`
    * Utilizes the **CLIP** (*Contrastive Language-Image Pre-Training*) model, pre-trained to understand semantic relationships between images and text.
* **Classification (Head):** `LogisticRegression` (Scikit-Learn)
    * A simple yet powerful linear model designed to handle high-dimensional feature vectors effectively.
* **Supporting Libraries:** `Transformers` (Hugging Face), `TensorFlow`, `Pandas`, `NumPy`, `Pillow`.

## ⚙️ Complete Workflow (Pipeline)

This project follows a systematic pipeline as described below:

### A. Data Preparation
1.  **Data Extraction:** The dataset is extracted from the source zip file (`kaggle-clash-4.zip`).
2.  **Strategic Splitting:**
    * The dataset is split into **Training Data (80%)** and **Validation Data (20%)**.
    * Implements a `stratify` strategy to ensure the proportion of classes (label 0 and 1) remains balanced across both sets, which is crucial for handling **imbalanced datasets**.

### B. Feature Extraction
This stage leverages the frozen CLIP model (`trainable = False`) as a dedicated *feature extractor*:
* **Preprocessing:** Automatically handles corrupt images by generating black placeholder images.
* **Text Embedding:** Converts input sentences into meaningful numerical vectors.
* **Image Embedding:** Converts input images into visual numerical vectors.

### C. Feature Engineering (Multimodal Fusion)
Instead of simply concatenating two features, this project implements an **Interaction Fusion** technique:
1.  **Text Features:** Raw vectors from the CLIP text encoder.
2.  **Image Features:** Raw vectors from the CLIP image encoder.
3.  **Interaction Features:** Generated via **element-wise multiplication** between text and image vectors.
    * *Objective:* To capture unique correlations where an offensive meaning might only emerge when specific text and images are paired together.
4.  **Concatenation:** All three vectors are combined into a single long vector with a dimension of **1536**.

### D. Model Training
Training is conducted using a Scikit-Learn Pipeline:
* **StandardScaler:** Normalizes data distribution to ensure uniform feature scaling.
* **Logistic Regression:**
    * `class_weight='balanced'`: Assigns a higher penalty to errors on the minority class to ensure fair modeling.
    * `C=0.01`: Strong regularization to prevent **overfitting** on high-dimensional features.

## 📊 Evaluation & Results
* **Method:** The model is evaluated on the 20% validation set.
* **Metrics:** Precision, Recall, and F1-Score (via `classification_report`).
* **Accuracy:** Achieved approximately **64%** on the validation data.

## 🌟 Key Advantages
1.  **Efficiency:** Does not require heavy *fine-tuning* (GPU-friendly) as it leverages CLIP's intrinsic pre-trained knowledge.
2.  **Robustness:** The **Multimodal Fusion** technique (interaction features) is proven effective in capturing complex nuances in memes.
3.  **Imbalance Handling:** The combination of *stratified split* and *class weighting* ensures stable performance even with uneven data distribution.

## 💻 How to Run
1.  Clone this repository.
2.  Ensure the dataset (`train.csv`, `test.csv`, and the `images/` folder) is available.
3.  Run the `iris_final (1).ipynb` notebook.
4.  Prediction results will be saved as `submission_MODEL_TERBAIK_0.60.csv`.
