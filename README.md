# Candidate Job Selection Prediction Model

A lightweight binary classification model trained from scratch using gradient descent with $L_2$ weight regularization to predict job applicant hiring decisions based on CGPA and work experience[cite: 1].

📄 **[Read the Full Research Paper (PDF)](./job_prediction_model_paper.pdf)**

---

## Abstract
Predicting job applicant hiring decisions based on quantitative performance metrics is a fundamental classification problem in automated recruitment systems[cite: 1]. This project implements a binary logistic regression model trained from scratch using gradient descent with $L_2$ weight regularization[cite: 1]. Using a dataset of 300 candidates, the model decouples training from inference to enable zero-dependency runtime predictions[cite: 1].

---

## Model & Formulation
The hiring probability is modeled using the logistic sigmoid function[cite: 1]:
$$z = w_0 + w_1 \cdot \text{CGPA} + w_2 \cdot \text{Experience}$$
$$p = \frac{1}{1 + e^{-z}}$$

### Learned Parameters
- **Bias ($w_0$):** `-8.1441` (Baseline threshold)[cite: 1]
- **CGPA Weight ($w_1$):** `+1.1485` (Strong positive influence)[cite: 1]
- **Experience Weight ($w_2$):** `+0.2916` (Moderate positive influence)[cite: 1]

### Linear Decision Boundary
$$\text{Experience} \approx -3.939 \cdot \text{CGPA} + 27.932$$[cite: 1]

---

## Repository Structure
- `AI_job_model3.py`: Custom training pipeline featuring regularized gradient descent and evaluation[cite: 2].
- `AI_job_getting_prediction.py`: Lightweight runtime inference script with zero external dependencies[cite: 1, 3].
- `job_applicants_300.csv`: Synthetic dataset used for model optimization[cite: 1, 2].
- `job_prediction_model_paper.pdf`: Full paper detailing mathematical derivations and experiment results[cite: 1].
