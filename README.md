# Chapter 6 – Logistic Regression: Exercise 1
## Topic Vibe Coding: Logistic Regression – Changing the Classification Threshold

This project implements **Chapter 6, Exercise 1** using the diabetes dataset. The main purpose is to build a Logistic Regression model and compare its predictions when the classification threshold is changed from `0.5` to `0.3`.

---

## 1. What are we predicting?
The model predicts whether a person has diabetes or not.
* **Outcome = 0** → No diabetes
* **Outcome = 1** → Diabetes

> ⚠️ **Note:** The model is not predicting gender or pregnancy. These are input features that help the model predict the final diabetes outcome.

## 2. What is the model doing?
The Logistic Regression model takes the input features and calculates a probability that the person belongs to class 1 (diabetes).

* **Threshold = 0.5**
  * Probability \(\ge\) 0.5 → Predict `1` (diabetes)
  * Probability < 0.5 → Predict `0` (no diabetes)
* **Threshold = 0.3**
  * Probability \(\ge\) 0.3 → Predict `1` (diabetes)
  * Probability < 0.3 → Predict `0` (no diabetes)

## 3. Why did we change 0.5 to 0.3?
Exercise 1 specifically requires changing the threshold from `0.5` to `0.3` to observe how the classification results shift. When the threshold is lowered, more cases are classified as positive (class 1). 

We compare the effect of this change on:
* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* ROC-AUC

## 4. Data Preparation
The dataset consists of:
* **8 feature columns:** Input health information.
* **1 binary label column (Outcome):** What we want to predict.

The code separates them into:
* `X` = features
* `y` = Outcome

## 5. Train-Test Split
The data is divided into:
* **80%** Training data
* **20%** Testing data
* `random_state = 5` (ensures the split is reproducible)

## 6. StandardScaler
Input features are standardized using `StandardScaler`. The scaler is fitted on the training data and then used to transform the test data to prevent data leakage.

## 7. Logistic Regression Model
The model initialized is:
```python
LogisticRegression(max_iter=1000)
```
The model learns patterns from the training data and evaluates them on the test data.

## 8. Why do we use `predict_proba()`?
To manually change the classification threshold, we extract the predicted probabilities for the positive class (class 1) instead of using the default hard assignments:
```python
y_probability = model.predict_proba(X_test_scaled)[:, 1]
```
We then apply our logic for both thresholds (`0.5` and `0.3`).

## 9. Evaluation Metrics
* **Accuracy:** How many predictions were correct overall?
* **Precision:** Out of all the people predicted as having diabetes, how many actually had diabetes? 
  \[\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}\]
* **Recall:** Out of all people who actually had diabetes, how many did the model correctly identify? 
  \[\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}\]
* **F1-score:** A harmonic mean that balances precision and recall.
* **ROC-AUC:** Measures how well the model separates the two classes using predicted probabilities.

## 10. Confusion Matrix Breakdown

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | TN (True Negative) | FP (False Positive) |
| **Actual 1** | FN (False Negative) | TP (True Positive) |

* **TN:** Actual 0, predicted 0.
* **FP:** Actual 0, predicted 1.
* **FN:** Actual 1, predicted 0.
* **TP:** Actual 1, predicted 1.

> **Easy way to remember Precision:** When the model says "diabetes", how often is it actually correct?

## 11. Human Checkpoints
Chapter 6 highlights 4 critical checkpoints:
1. Confirm that the target variable is binary.
2. Check class imbalance before trusting accuracy.
3. Understand precision, recall, F1-score, and the confusion matrix.
4. Decide whether the classification threshold needs adjustment.

## 12. Why don't we use only accuracy?
Accuracy can be misleading if the dataset has class imbalance. Checking **Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC** provides a complete picture of the classifier's performance.

---

## 13. Questions Asked (Viva Prep)

### Q1. What are you predicting?
**Answer:** I am predicting whether a person has diabetes or not (`0` = No diabetes, `1` = Diabetes).

### Q2. What is the target variable?
**Answer:** The target variable is `Outcome`.

### Q3. What is the role of pregnancy/pregnancies?
**Answer:** `Pregnancies` is an input feature used by the model to help make its prediction. It is not the target variable.

### Q4. What is the role of values like 88 and 35?
**Answer:** They are specific data values belonging to the input patient features. The model processes these values collectively to generate the output probability.

### Q5. What is precision?
**Answer:** Precision tells us, out of all the individuals the model flagged as having diabetes, how many truly had it. 
\[\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}\]

### Q6. Why don't we take only accuracy?
**Answer:** Accuracy measures overall correctness but fails to reveal how well the model handles specific classes (especially in imbalanced data). We check precision, recall, F1-score, confusion matrix, and ROC-AUC for a thorough validation.

### Q7. What are the Human Checkpoints?
**Answer:** Check if the target is binary, investigate class imbalance, fully comprehend evaluation metrics, and determine if the decision threshold needs to be adjusted.

### Q8. Why did you change 0.5 to 0.3?
**Answer:** The exercise requirements explicitly ask us to change the threshold from `0.5` to `0.3` to observe how classification metrics and error balances alter.

### Q9. Why do you use `predict_proba()`?
**Answer:** Because the standard `.predict()` function forces a default 0.5 threshold. Using `.predict_proba()` gives us raw probabilities so we can apply custom thresholds like `0.3`.

### Q10. What happens when the threshold is lowered?
**Answer:** A lower threshold makes the model more sensitive. It becomes easier to classify a sample as positive (`1`), which typically increases **Recall** but might drop **Precision**, changes the confusion matrix distribution, and can alter overall accuracy.

---

## 14. Short Viva Explanation
> "I used the diabetes dataset to build a Logistic Regression model. The target is `Outcome`, where `0` means no diabetes and `1` means diabetes. I split the data into 80% training and 20% testing, scaled the features using `StandardScaler`, and trained the model. Then, I used `predict_proba` to capture raw probabilities. As required by Exercise 1, I compared the classification changes between a `0.5` and a `0.3` threshold across accuracy, precision, recall, F1-score, ROC-AUC, and the confusion matrix."
