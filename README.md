# 🕵️‍♂️ Credit Card Fraud Detection

This project focuses on building and evaluating machine learning models to detect fraudulent credit card transactions using advanced data preprocessing, feature engineering, and classification techniques. 

Two primary models are trained and compared:
- Logistic Regression with ElasticNet penalty
- LightGBM (Gradient Boosted Trees)

---

## 📂 Project Structure

```text
credit-card-fraud-detection/
│
├── data/                      # Raw dataset (not included in repo)
├── outputs/                   # Model predictions, plots, metrics
│   ├── plots/                 # Visualizations of metrics
│   └── predictions/           # CSV files of predicted labels
│
├── models/                    # Trained and saved models (pkl)
├── src/                       # Source code
│   ├── data_preprocessing.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── train.py
│   ├── utils.py
│   └── visualization.py
│
├── tests/                     # Testing model inference
│   └── test_model.py
│
├── main.py                    # Main execution script
└── README.md
```


---

## 🔍 Dataset

The dataset used is from the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud), which contains anonymized features extracted from real European cardholder transactions in September 2013.

- **Size**: 284,807 transactions
- **Fraudulent**: ~0.17% of data

---

## ⚙️ Key Features

- ✅ Feature scaling and transformation using `ColumnTransformer`
- ✅ Imbalanced data handling via `SMOTE`
- ✅ Model tuning using `precision-recall` optimization
- ✅ Visualizations for performance metrics
- ✅ Threshold selection based on desired precision

---

## 🧪 Models & Thresholding

Models are trained using pipelines with transformations + SMOTE and evaluated using:
- **Accuracy**
- **F1 Score**
- **Precision & Recall**
- **ROC AUC**
- **PR Curve**

Thresholds are tuned to meet:
- **75% precision for Logistic Regression**
- **90% precision for LightGBM**

---

## 📈 Visualizations

Generated plots include:
- ROC curves
- Precision-Recall curves
- Accuracy & F1 bar plots
- Confusion matrices

All outputs are saved in `outputs/plots`.

---

## ▶️ How to Run

1. **Clone the repo**  
```bash
git clone https://github.com/mjavadzare/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```
2. **Install dependencies**
```bash
pip install -r requirements.txt
```
3. **Download the dataset from Kaggle and place it inside data/raw/.**
4. **Run main.py**

---
## 📦 Dependencies

scikit-learn

imbalanced-learn

lightgbm

numpy

pandas

matplotlib

---

## 📬 Contact

Developed by [Mohammad Javad Zare](https://github.com/mjavadzare)

If you have any questions, suggestions, or feedback, feel free to:

- 📂 [Open an Issue](https://github.com/mjavadzare/credit-card-fraud-detection/issues)
- 🍴 [Fork the Project](https://github.com/mjavadzare/credit-card-fraud-detection/fork)

Your contributions and feedback are always welcome!


---

## 📜 License

This project is licensed under the MIT License.
