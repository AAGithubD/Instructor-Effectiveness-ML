# 📚 Instructor Effectiveness Machine Learning Project

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning pipeline built for an **EdTech Platform** to quantify, evaluate, and predict **Instructor Effectiveness Tiers** (`High`, `Medium`, `Low`) based on student outcome performance, engagement rates, and survey feedback metrics.

---

## 📌 Table of Contents
- [Business Problem & Context](#-business-problem--context)
- [Dataset Overview](#-dataset-overview)
- [Project Architecture & Workflow](#-project-architecture--workflow)
- [Model Benchmarking & Results](#-model-benchmarking--results)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [How to Run & Predict](#-how-to-run--predict)
- [Key Business Insights & Governance](#-key-business-insights--governance)
- [License](#-license)

---

## 🎯 Business Problem & Context

In online education platforms, evaluating instructor quality strictly through raw student ratings or individual batch completion rates leads to significant bias due to course difficulty, batch size, and student engagement variations.

### **Key Objectives:**
1. **Aggregate** batch-level records into unique, instructor-level profiles.
2. **Formulate** a domain-justified composite **Instructor Effectiveness Score** and categorize instructors into three performance tiers (`High`, `Medium`, `Low`).
3. **Train & Benchmark** multiple Machine Learning classification algorithms to identify top drivers of teaching effectiveness.
4. **Deploy** serialized model artifacts (`.pkl`) for continuous inference and internal coaching diagnostics.

---

## 📊 Dataset Overview

The project processes **2,000 course batch records** aggregated across **120 unique instructors**:

| Category | Feature Name | Description |
| :--- | :--- | :--- |
| **Identifiers** | `batch_id`, `instructor_id`, `course_id` | Unique identifiers for batches, teachers, and courses. |
| **Outcomes** | `completion_rate`, `dropout_rate` | Student completion ratio and churn percentage (0.0 to 1.0). |
| | `avg_score_improvement` | Grade growth between pre/post assessments. |
| | `avg_quiz_score` | Batch average quiz performance. |
| **Engagement** | `avg_watch_time` | Normalized lecture video consumption time. |
| | `assignment_submission_rate` | Ratio of assignments submitted by students. |
| | `forum_activity_rate` | Discussion forum participation rate. |
| **Feedback** | `avg_feedback_score` | Student satisfaction rating (1.0 to 5.0). |
| | `feedback_response_rate` | Survey response rate per batch. |

---

## 🔄 Project Architecture & Workflow

### **Phase 1: Exploratory Data Analysis & Aggregation**
* Verified data integrity (0 missing values, 0 duplicate records across 2,000 batch rows).
* Aggregated batch-level data by `instructor_id` to generate **120 unique instructor profiles** (`instructor_effectiveness_dataset.csv`).
* Engineered a workload feature (`total_batches`) using count aggregation.

### **Phase 2: Target Engineering**
* Formulated a composite **Instructor Effectiveness Score** ($S$) combining learning outcomes, engagement, and feedback:
  $$S = 0.25 \cdot \text{completion\_rate} - 0.20 \cdot \text{dropout\_rate} + 0.20 \cdot \text{avg\_score\_improvement} + 0.15 \cdot \text{avg\_quiz\_score} + 0.10 \cdot \text{avg\_watch\_time} + 0.10 \cdot \text{assignment\_submission\_rate} + 0.10 \cdot \text{avg\_feedback\_score} + 0.05 \cdot \text{forum\_activity\_rate} + 0.05 \cdot \text{feedback\_response\_rate}$$
* Categorized instructors into three balanced tiers using 33rd and 66th percentiles:
  * 🔴 **Low Tier** ($N=40$)
  * 🟡 **Medium Tier** ($N=39$)
  * 🟢 **High Tier** ($N=41$)

### **Phase 3: Preprocessing & Leakage Prevention**
* Partitioned data into an **80/20 Stratified Train-Test Split** ($N_{\text{train}}=96$, $N_{\text{test}}=24$).
* Scaled features using `StandardScaler` **fitted strictly on training data** to avoid data leakage.

### **Phase 4: Model Training & Evaluation**
* Benchmarked 6 algorithms: Logistic Regression, Decision Tree, Random Forest, KNN, SVM, and Naive Bayes.
* **Random Forest Classifier** achieved the top score: **95.8% Accuracy** and **0.958 F1-Score**.

---

## 🏆 Model Benchmarking & Results

Evaluation conducted on holdout test set ($N=24$):

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | **F1-Score (Weighted)** |
| :--- | :---: | :---: | :---: | :---: |
| 🥇 **Random Forest Classifier** | **95.8%** | **0.963** | **0.958** | **0.958** |
| 🥈 Decision Tree Classifier | 91.7% | 0.933 | 0.917 | 0.915 |
| 🥉 Logistic Regression | 91.7% | 0.926 | 0.917 | 0.913 |
| 4️⃣ Support Vector Machine (SVM) | 83.3% | 0.832 | 0.833 | 0.830 |
| 5️⃣ $k$-Nearest Neighbors ($k$-NN) | 83.3% | 0.832 | 0.833 | 0.830 |
| 6️⃣ Gaussian Naive Bayes | 79.2% | 0.837 | 0.792 | 0.794 |

---

## 📂 Repository Structure

```text
Instructor Effectiveness(Machine Learning)/
│
├── data/
│   ├── raw/
│   │   └── README.md                                # Details regarding raw source data
│   └── processed/
│       └── instructor_effectiveness_dataset.csv      # Aggregated instructor-level dataset
│
├── docs/
│   └── project_workflow.txt                         # Detailed technical workflow description
│
├── models/
│   ├── random_forest_model.pkl                      # Serialized trained Random Forest model
│   └── scaler.pkl                                   # Serialized StandardScaler object
│
├── notebooks/
│   └── Instructor Effectiveness Modeling.ipynb      # Main Jupyter notebook with code & output
│
└── README.md                                        # Project documentation