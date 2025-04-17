# 💘 Tinder Match Prediction using Machine Learning

A fun and practical machine learning project that predicts whether two Tinder users will match based on their profile characteristics like age, interests, proximity, and perceived attractiveness.

> ⚠️ Disclaimer: This is a hypothetical project using synthetic data. No real user data is used.

---

## 📊 Project Overview

The goal of this project is to build a binary classification model that predicts the likelihood of a Tinder match based on:
- Age differences
- Common interests
- Location distance
- Attractiveness rating (1–10)

The dataset is synthetically generated to simulate realistic matching behavior.

---

## 🧠 Machine Learning Pipeline

1. **Data Generation**
   - Randomly generates user data (age, distance, interests, attractiveness).
   - A basic rule-based function simulates whether a match occurs.

2. **Data Preprocessing**
   - Features and labels are split.
   - Dataset is divided into training and test sets.

3. **Model Building**
   - A `RandomForestClassifier` is trained to learn patterns in the data.

4. **Evaluation**
   - Model is evaluated using accuracy, confusion matrix, and classification report.

---

## 🔍 Example Features

| Feature               | Description                                 |
|-----------------------|---------------------------------------------|
| `user1_age`           | Age of the first user                       |
| `user2_age`           | Age of the second user                      |
| `distance_km`         | Distance between users (in kilometers)      |
| `common_interests`    | Number of shared interests                  |
| `user1_attractiveness` | Attractiveness rating (1-10)              |
| `user2_attractiveness` | Attractiveness rating (1-10)              |
| `match`               | Target variable (1 = Match, 0 = No Match)   |

---

## 🧪 Results

The Random Forest classifier achieved an average accuracy of **~88%** on the synthetic test data.

```
Accuracy: 0.88

Confusion Matrix:
[[124  10]
 [ 14  52]]

Precision/Recall:
    0 = No Match → Precision: 0.90, Recall: 0.93
    1 = Match    → Precision: 0.84, Recall: 0.79
```

---

## 🛠 Tech Stack

- Python 🐍
- Pandas, NumPy
- scikit-learn
- Seaborn, Matplotlib

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Max-Ola/Tinder-Matching-Prediction
cd tinder-match-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the model
```bash
python tinder_match_predictor.py
```

---

## 📁 File Structure

```
📦 tinder-match-predictor
├── README.md
├── requirements.txt
├── tinder_match_predictor.py
└── data/
    └── generated_data.csv (optional if you save synthetic data)
```

---

## 💡 Future Improvements

- Integrate NLP to analyze user bios
- Add swipe history or user preferences
- Build a Streamlit dashboard for live demos
- Try different classifiers (XGBoost, Neural Networks)

---

## 📸 Demo

<img src="demo-screenshot.png" alt="Demo Screenshot" width="500"/>

---

## 👨‍💻 Author

**Maxwel Olande**  
DevOps | Cloud | ML Hobbyist  
[LinkedIn](https://linkedin.com/in/maxwelolande) | [Twitter](https://twitter.com/MaxUrus254)

---

## 📜 License

MIT License. Use it, remix it, have fun.

---
