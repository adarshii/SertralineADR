# 🧬 AI-Driven Adverse Drug Reaction (ADR) Prediction System

An explainable, interactive clinical decision-support system for predicting
**adverse drug reaction (ADR) risk** associated with **Sertraline**, integrating
clinical, pharmacological, and multi-omics features using machine learning.

This system is designed for **research, educational, and pharmacovigilance use**
and provides transparent risk interpretation through visual analytics and an
AI-assisted explanation interface.

---

## 🚀 Key Features

- **Machine-learning–based ADR risk prediction**
- **ADR signal strength (%)** with risk categorization
- **Model confidence estimation** (uncertainty awareness)
- **Organ-specific risk visualization** (radar plots)
- **Time-dependent ADR risk timeline**
- **SHAP-based explainability** for feature contributions
- **Interactive AI assistant** for clinicians & researchers
- **“Explain this graph”** buttons with visual highlighting
- **Hover-based contextual explanations**
- **User feedback scoring** for explanation usefulness
- **Streamlit Cloud deployment ready**

---

## 🖥️ Live Demo

👉 *(Add your Streamlit Cloud URL here once deployed)*  
Example:  
`https://your-app-name.streamlit.app`

---

## 🧠 Intended Users

- **Clinicians** – to interpret ADR risk and uncertainty  
- **Researchers** – to explore model behavior and biological drivers  
- **Pharmacovigilance analysts** – to support post-marketing safety analysis  

---

## 📊 Methodology Overview

### 🔹 Model
- **Algorithm:** LightGBM classifier  
- **Output:** ADR signal score (probability-based)  
- **Risk interpretation:** Low / Moderate / High  

### 🔹 Input Features
- Clinical factors (age, dose, comorbidities)
- Pharmacological variables
- Multi-omics features (genomics / proteomics / pathway proxies)

### 🔹 Explainability
- **SHAP values** for global and local feature importance
- Natural-language interpretation via AI assistant

### 🔹 Confidence Estimation
- Computed as distance from the model decision boundary (0.5)
- Expressed as **High / Moderate / Low confidence**

---

## 🤖 AI Clinical & Research Assistant

The integrated AI assistant provides **interpretive support**, not medical advice.

It can explain:
- What the prediction means
- How confident the model is
- Why certain features increased risk
- How to interpret SHAP plots
- What each visualization represents

The assistant is **context-aware**, responding differently depending on:
- User role (Clinician vs Researcher)
- Active visualization or tab
- Model outputs for the current patient profile

---

## 📁 Project Structure
Sertraline/
├── app.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── styles.css # Custom UI styling
├── assets/ # Images, logos, screenshots
│ └── logo.png
├── models/ # Trained ML models
│ └── model.pkl
├── utils/ # Helper modules (optional refactor)
│ ├── model_utils.py
│ ├── confidence_utils.py
│ ├── explain_utils.py
│ ├── plot_utils.py
│ └── db_utils.py
├── data/ # Reference data (if any)
└── README.md # Project documentation


## ⚙️ Installation & Local Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

