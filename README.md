# ❤️ Heart Disease Prediction – Scikit-Learn vs PyTorch

## 📌 Project Overview

This project predicts the presence of heart disease using both:

- Scikit-learn (Traditional Machine Learning)
- PyTorch (Deep Learning)

The objective is to compare performance between classical ML models and a neural network model on structured medical (tabular) data.

---

## 🎯 Problem Statement

Given clinical parameters about a patient (age, cholesterol, chest pain type, blood pressure, etc.),  
can we predict whether the patient has heart disease?

This is a binary classification problem.

---

## 📂 Project Structure

```
Heart-Disease-Prediction-ML-PyTorch-Sklearn/
│
├── sklearn_model/
│   ├── 01_Data_Preprocessing.ipynb
│   ├── 02_Model_Training_Sklearn.ipynb
│   └── 03_Model_Evaluation.ipynb
│
├── pytorch_model/
│   ├── 01_Data_Preprocessing.ipynb
│   ├── 02_Model_Training_Pytorch.ipynb
│   └── 03_Model_Evaluation.ipynb
│
├── data/
│   └── heart.csv
│
└── README.md
```

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- PyTorch
- Jupyter Notebook

---

## 🔵 Scikit-learn Workflow

### 1️⃣ Data Preprocessing
- Handling missing values
- Feature selection
- Train-test split
- Feature scaling

### 2️⃣ Model Training
- Logistic Regression / Random Forest (based on your implementation)
- Model fitting on training data

### 3️⃣ Model Evaluation
- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curve

---

## 🔴 PyTorch Workflow

### 1️⃣ Data Preprocessing
- Same preprocessing steps for fair comparison
- Conversion to tensors

### 2️⃣ Model Training
- Custom Neural Network using `nn.Module`
- Fully connected layers
- ReLU activation
- Sigmoid output layer
- Binary Cross Entropy Loss
- Adam optimizer

### 3️⃣ Model Evaluation
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Comparison with Scikit-learn

---

## 📊 Model Comparison

Both approaches are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

### 🔎 Observation

For structured tabular datasets, traditional machine learning models often perform equally or better than deep learning models unless the dataset is very large.

However, implementing PyTorch demonstrates deep learning knowledge and flexibility.

---

## ▶️ How to Run

### 1. Clone Repository
```
git clone https://github.com/thepratikmore/Heart-Disease-Prediction-ML-PyTorch-Sklearn.git
cd Heart-Disease-Prediction-ML-PyTorch-Sklearn
```

### 2. Install Required Libraries
```
pip install numpy pandas matplotlib seaborn scikit-learn torch
```

### 3. Run Notebooks
Open Jupyter Notebook:

```
jupyter notebook
```

Run notebooks in order:

For Scikit-learn:
1. 01_Data_Preprocessing
2. 02_Model_Training_Sklearn
3. 03_Model_Evaluation

For PyTorch:
1. 01_Data_Preprocessing
2. 02_Model_Training_Pytorch
3. 03_Model_Evaluation

---

## 🚀 Future Improvements

- Hyperparameter tuning
- Cross-validation
- Model deployment (Flask / FastAPI)
- Add UI for prediction

---

## 📌 Key Learnings

- End-to-end ML workflow
- Difference between classical ML and deep learning
- Model evaluation techniques
- Structured project organization
- Comparative performance analysis

---

## ⚠️ Disclaimer

This project is for educational purposes only and not intended for medical diagnosis.

---

## 👨‍💻 Author

Pratik More  
Aspiring Machine Learning Engineer
