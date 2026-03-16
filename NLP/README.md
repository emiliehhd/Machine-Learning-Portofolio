# Natural Language Processing (NLP)

This directory contains a progression of projects focusing on Text Mining and Sentiment Analysis. The curriculum starts with foundational preprocessing and moves toward sequence modeling with Recurrent Neural Networks (RNNs).

## 📂 Contents

### 0. [Data Preprocessing & Exploration (EDA)](./01_Preprocessing_EDA.ipynb)
**Goal:** Preprocess the data and create intermediate data visualization.
* **Key Tasks:** Identifying and handling outliers, Handling missing values, Visualization.
* **Evaluation:** Interquartile Range (IQR) method, Boxplot, and Correlation Heatmap.
* **Tools:** `Pandas`, `Matplotlib`, `Seaborn`.

### 1. 
**Goal:** Master the art of cleaning raw text data before feeding it into machine learning models.
* **Key Tasks:** Load data, Stop-word removal, Lemmatization, and Stemming.
* **Analysis:** Word frequency distributions, N-grams analysis, and WordClouds.
* **Tools:** `NLTK`, `SpaCy`, `Matplotlib`.

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
2. Install the specific NLP requirements:
   ```bash
   pip install nltk spacy scikit-learn torch
   python -m spacy download en_core_web_sm
