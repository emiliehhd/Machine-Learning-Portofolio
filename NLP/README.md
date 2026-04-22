# Natural Language Processing (NLP)

This directory contains a progression of projects focusing on Text Mining and Sentiment Analysis. The curriculum starts with foundational preprocessing and moves toward sequence modeling with Recurrent Neural Networks (RNNs).

## 📂 Contents

### 1. [Data Preprocessing & Exploration (EDA)](./01_Preprocessing_EDA.ipynb)
**Goal:** Predict wine quality based on physico-chemical properties through exploratory data analysis (EDA) and statistical examination.
* **Key Tasks:** Load and explore data, Clean data, Distribution and correlation analysis.
* **Algorithms and Methods:** Interquartile Range (IQR) method, Multidimensional visualization.
* **Tools:** `Pandas`, `Matplotlib`, `Seaborn`.
* [Execute notebook on Google Colab](https://colab.research.google.com/github/emiliehhd/Machine-Learning-Portofolio/blob/main/NLP/01_Preprocessing_EDA.ipynb)


### 2. [Movie Review Classification (Baseline Models)](./02_Movie_Review_Classification.ipynb)
**Goal:** Build a robust sentiment analysis baseline using traditional Machine Learning.
* **Dataset:** IMDB / Movie Reviews dataset.
* **Features:** TF-IDF Vectorization and Bag-of-Words (BoW).
* **Algorithms:** Logistic Regression and Support Vector Machines (SVM).
* **Evaluation:** Confusion Matrix, Precision-Recall curves, and F1-Score.
* **Tools:** `Scikit-learn`, `Pandas`.

### 3. [Sentiment Analysis with RNNs](./03_RNN_Sentiment_Analysis.ipynb)
**Goal:** Implement Deep Learning architectures to capture sequential dependencies in text.
* **Model Architecture:** Simple RNNs and LSTMs (Long Short-Term Memory).
* **Key Concepts:** Word Embeddings (`Word2Vec` / `GloVe`), Padding, and Sequence Masking.
* **Results:** Comparing the performance of sequential models against the statistical baselines from Notebook 2.
* **Tools:** `PyTorch` (or `TensorFlow/Keras`), `NumPy`.

---

## Key Learnings
* Understanding the impact of **text normalization** on model accuracy.
* Handling the **sparsity** of TF-IDF matrices vs. the **density** of word embeddings.
* Managing **vanishing gradients** in standard RNNs by using LSTM cells.

## How to Run
Most notebooks can be opened directly in Google Colab. For local execution:
1. Ensure you have the datasets (or links provided in the notebooks).
2. Install the requirements
