
# **EEG Emotion Classification**

This project focuses on classifying emotions (Negative, Neutral, Positive) using EEG brainwave data. The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions/data). The project involves preprocessing the data, training machine learning models, and building an LSTM-based deep learning model to classify emotions effectively.

---

## **Table of Contents**
1. [Dataset](#dataset)
2. [Project Structure](#project-structure)
3. [Setup and Installation](#setup-and-installation)
4. [Usage](#usage)
5. [Methods](#methods)
6. [Results](#results)
7. [Contributions](#contributions)

---

## **Dataset**
- The dataset contains EEG brainwave signals and their corresponding emotional labels (Negative, Neutral, Positive).
- Dataset source: [Kaggle - EEG Brainwave Dataset: Feeling Emotions](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions/data).

### **Features**
- EEG signal data (columns with mean, fft, etc.).
- Target labels: `NEGATIVE`, `NEUTRAL`, and `POSITIVE`.

---

## **Setup and Installation**
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ivandezra/eeg-emotion-classification.git
   cd eeg-emotion-classification
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.9+ installed. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset:**
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions/data).
   - Place the `emotions.csv` file in the `data/` directory.

4. **Run the Project:**
   Execute the `main.py` file to train and evaluate the models:
   ```bash
   python main.py
   ```

---

## **Usage**
1. **Data Preprocessing:**
   - Data is standardized and split into training and testing sets.
2. **Training Models:**
   - Train various machine learning models (Naive Bayes, SVM, Logistic Regression, Decision Tree, Random Forest).
   - Train an LSTM deep learning model for improved results on sequential data.
3. **Evaluation:**
   - Evaluate models using precision, recall, F1-score, and confusion matrix.

---

## **Methods**
### **1. Machine Learning Models**
- **Naive Bayes:** Assumes features are independent; suitable for simple tasks.
- **Logistic Regression:** Linear classifier with regularization.
- **Decision Tree:** Tree-based model for decision-making.
- **Random Forest:** Ensemble model combining multiple decision trees.

### **2. LSTM Model**
- Long Short-Term Memory (LSTM) networks capture temporal dependencies in EEG signals.
- Architecture:
  - LSTM layers for sequence modeling.
  - Dropout layers to prevent overfitting.
  - Dense layers for classification.

### **3. Evaluation Metrics**
- **Classification Report:** Precision, recall, and F1-score.
- **Confusion Matrix:** Visual representation of true vs. predicted classes.

---

## **Results**
- **Machine Learning Models:**
  - Random Forest performed the best among sklearn models with hyperparameter tuning.
- **LSTM Model:**
  - Achieved higher accuracy (~98%) by leveraging sequential dependencies in EEG data.

---

## **Contributions**
- **Developer:** [Iván Hernández Ramos](https://linkedin.com/in/ivanhernandezramos/).
- **Dataset:** [Kaggle - EEG Brainwave Dataset](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions/data).

---
