# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

---

## 🧠 Model Details
- **Model type:** RandomForestClassifier  
- **Framework:** scikit-learn  
- **Trained using:** Python 3.13  
- **Owner:** Dhanush Kumar S. B.  
- **Developed for:** Accenture AI/ML Pre-Training Project (Census Income Prediction)

---

## 🎯 Intended Use
Predict whether a person earns **more than $50K/year** or **less than or equal to $50K/year** based on U.S. Census data.

Use cases include:
- Income classification
- Understanding factors influencing income
- Learning model deployment workflow (educational purpose)

Not for production or real-world financial decisions.

---

## 🧾 Training Data
- **Dataset:** U.S. Census Income dataset (census.csv)
- **Size:** ~32,000 samples
- **Features:** 14 attributes (age, workclass, education, occupation, etc.)
- **Target variable:** salary (">50K" or "<=50K")

---

## 🧪 Evaluation Data
- 20% of data used as test split.
- Same preprocessing and categorical encoding applied.

---

## 📊 Metrics
| Metric | Score |
|---------|-------|
| Precision | 0.743 |
| Recall | 0.637 |
| F1-score | 0.686 |

---

## ⚖️ Ethical Considerations
- The model is trained on historical census data, which may contain societal biases (e.g., gender or race correlations with income).  
- The model should **not be used** for employment, credit, or policy decisions.  
- Educational purpose only.

---

## ⚠️ Caveats and Recommendations
- Model performance can vary for underrepresented groups.
- Future improvement: try balanced sampling or bias mitigation techniques.
- Retraining recommended with more recent or representative data.
