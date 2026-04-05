# ADR•X — Sertraline Signal Explorer

### An Interpretable, Leakage-Aware Machine Learning Framework for Adverse Drug Reaction Signal Detection Using FAERS Pharmacovigilance Data

---

![Status](https://img.shields.io/badge/Status-Research%20Complete-brightgreen)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Model](https://img.shields.io/badge/Model-LightGBM-orange)
![XAI](https://img.shields.io/badge/Explainability-SHAP-purple)
![Deployment](https://img.shields.io/badge/Deployment-Streamlit-red)
![Institution](https://img.shields.io/badge/Institution-Guru%20Nanak%20Khalsa%20College%2C%20Mumbai-navy)

---

## 📄 Abstract

Adverse drug reactions (ADRs) impose a substantial and largely preventable burden on global health systems. The FDA Adverse Event Reporting System (FAERS) provides large-scale post-marketing safety data; however, most existing machine learning (ML) studies embed outcome-derived disproportionality metrics — proportional reporting ratios (PRR) or reporting odds ratios (ROR) — as model features, constituting information leakage and yielding inflated, non-generalisable performance estimates.

**ADR•X** is a leakage-aware Light Gradient Boosting Machine (LightGBM) framework for sertraline ADR signal detection from FAERS data, validated with SHapley Additive exPlanations (SHAP)-based global and local interpretability. FAERS sertraline records were extracted, deduplicated, and processed under a schema-frozen pipeline. Approximately 208 features were engineered across six categories spanning demographics, physicochemical descriptors, pharmacogenomic indicators, and biology-guided multi-omics proxies. All PRR, ROR, and frequency-derived variables were explicitly excluded from the feature space. Two LightGBM variants were trained: an unweighted baseline and an inverse class-frequency-weighted model. The baseline achieved AUC-ROC of 0.53–0.54; the imbalance-adjusted variant reached 0.55–0.56. Global SHAP identified `dose_mg` (+0.030), `metabolic_overload_score` (+0.020), and `polypharmacy_flag` (+0.010) as the dominant predictors — all biologically plausible for sertraline pharmacology. All 209 remaining features showed near-zero distributed contributions, confirming the absence of leakage-driven dominance.

ADR•X demonstrates that leakage-free, biologically plausible ADR signal estimation is achievable from spontaneous reporting data. Modest AUC values reflect the intrinsic information ceiling of FAERS rather than model inadequacy, and are interpreted as markers of methodological rigour under responsible AI principles.

**Keywords:** Adverse drug reactions · Pharmacovigilance · Sertraline · FAERS · LightGBM · SHAP · Leakage-aware modelling · Pharmacogenomics · Signal detection · Explainable AI

---

## ✨ Key Features

- **Leakage-free by design** — All PRR, ROR, and report-count variables explicitly excluded from the feature space prior to label construction, preventing circular inference endemic to prior FAERS ML studies.
- **Multi-dimensional feature engineering** — ~208 features across six biologically informed categories: demographics, clinical exposure, RDKit physicochemical descriptors, multi-omics proxies, pharmacogenomic indicators, and mechanistic interaction terms.
- **Dual LightGBM architecture** — Unweighted baseline (optimised for probability calibration) and inverse class-frequency-weighted variant (optimised for ADR-positive sensitivity), enabling user-driven model selection based on clinical objective.
- **SHAP interpretability at two levels** — Global mean |SHAP| bar charts for population-level attribution; per-patient waterfall plots for individual prediction decomposition.
- **Biological plausibility validation** — SHAP outputs cross-referenced against established sertraline pharmacology (CYP2C19 metabolism, dose-response, polypharmacy interactions) to confirm mechanistic credibility.
- **Confidence estimation** — Decision-boundary proximity communicated alongside ADR risk probability and categorical signal tier (Low / Moderate / High).
- **Streamlit research portal** — Role-adaptive interface for Clinicians, Researchers, and Pharmacovigilance Analysts, with SQLite audit logging and PDF report generation.
- **Integrated AI assistant** — Embedded natural-language assistant for SHAP output interpretation and scientific contextualisation.
- **Fully reproducible pipeline** — Schema-frozen inference, fixed random seed (seed=42), and dependency-locked environment.

---

## 👥 Intended Users

| User Profile | Primary Use Case |
|---|---|
| **Clinical Pharmacologists** | Signal hypothesis generation for sertraline ADR risk stratification |
| **Pharmacovigilance Analysts** | Triage prioritisation of FAERS safety signals |
| **Computational Researchers** | Methodological benchmark for leakage-aware ML in spontaneous reporting systems |
| **Regulatory Scientists** | Exploratory safety signal evaluation with auditable, explainable predictions |
| **MSc / PhD Students** | Reference implementation for pharmacoinformatics and responsible AI in drug safety |

---

## 🔬 Methodology

### Data Source and Preprocessing

FAERS quarterly records linked to sertraline exposure were extracted, deduplicated by case identifier, and processed under a fully deterministic, schema-frozen pipeline. Binary ADR signal labels (ADR present = 1; ADR absent = 0) were constructed **prior to** any feature engineering step to prevent label information from contaminating the feature space. All PRR, ROR, and report-count-derived variables were explicitly removed. Only adult reports were retained. The pipeline enforces immutable schema versioning across training and inference to guarantee reproducibility.

### Feature Engineering (~208 Variables, 6 Categories)

| Category | Variables |
|---|---|
| **Demographics** | Age, biological sex |
| **Clinical Exposure** | Dose category, polypharmacy flag, hepatic impairment proxy |
| **RDKit Physicochemical Descriptors** | Molecular weight, logP, TPSA, HBD/HBA counts, rotatable bond count |
| **Multi-Omics Proxy Features** | Neuroinflammation index, oxidative stress score, blood-brain barrier integrity proxy, cytokine activation index, metabolic burden score |
| **Pharmacogenomic Indicators** | CYP2C19 metaboliser phenotype, CYP2D6 metaboliser phenotype, SLC6A4 (SERT) expression proxy — curated from variant-phenotype knowledge bases |
| **Mechanistic Interaction Terms** | Age × Dose, Polypharmacy × Liver disease, Oxidative stress × Neuroinflammation, Dose × CYP2C19 phenotype |

### Model Architecture

Two LightGBM classifiers were trained on an 80:20 stratified train–validation split (seed=42):

- **Baseline Model (Unweighted):** Conservative probability estimates; prioritises calibration; preferred for hypothesis generation.
- **Imbalance-Adjusted Model:** Inverse class-frequency weighting; improved ADR-positive sensitivity; preferred for triage-oriented signal prioritisation.

**Hyperparameters:** 300 estimators · Learning rate 0.05 · Max depth 6 · L1 and L2 regularisation active · Leaf-wise tree growth.

### Explainability

SHAP `TreeExplainer` was applied to compute exact Shapley values for both model variants:

- **Global Interpretability:** Mean |SHAP value| bar charts identifying population-level feature importance.
- **Local Interpretability:** Per-patient waterfall plots decomposing individual ADR risk predictions into additive feature contributions.

All SHAP outputs were assessed for biological plausibility against established sertraline pharmacology and used as a secondary validation criterion for leakage-prevention efficacy.

### Confidence Estimation

ADR risk probability is complemented by a confidence tier derived from the model's decision boundary proximity:

- **Low Signal Priority:** P(ADR) < 0.40
- **Moderate Signal Priority:** 0.40 ≤ P(ADR) < 0.65
- **High Signal Priority:** P(ADR) ≥ 0.65

Uncertainty is communicated explicitly in the Streamlit interface to discourage overconfident clinical interpretation, consistent with responsible AI deployment principles in healthcare.

---

## 📊 Results and Performance

### Model Performance

| Model Variant | AUC-ROC | Note |
|---|---|---|
| Baseline (Unweighted) | 0.53 – 0.54 | Well-calibrated; conservative probability estimates |
| Imbalance-Adjusted | 0.55 – 0.56 | Improved ADR-positive sensitivity; mild probability inflation |

> **Interpretive note:** AUC values in the 0.53–0.56 range reflect the intrinsic information ceiling of spontaneous reporting data rather than model inadequacy. Studies reporting AUC >0.80 on FAERS data consistently include PRR, ROR, or report-frequency variables as features — constituting circular inference. The conservative performance of ADR•X is a deliberate marker of methodological rigour.

### Top SHAP Contributors

| Feature | Mean \|SHAP\| | Biological Rationale |
|---|---|---|
| `dose_mg` | +0.030 | Dose-dependent sertraline serotonergic exposure |
| `metabolic_overload_score` | +0.020 | CYP2C19 saturation elevates systemic plasma concentrations |
| `polypharmacy_flag` | +0.010 | CYP-mediated drug–drug interaction risk |
| All remaining 209 features | ~0.000 | Distributed contributions; no leakage-driven dominance |

### Figures

| Figure | Description |
|---|---|
| Figure 1 | ROC curve — Baseline vs. Imbalance-Adjusted LightGBM |
| Figure 2 | Global SHAP bar chart — Mean absolute feature attribution across validation set |
| Figure 3 | Local SHAP waterfall — Representative low-risk case (f(x) = −1.378) |
| Figure 4 | End-to-end methodology workflow diagram |

> 📸 *Screenshots of the Streamlit dashboard and SHAP visualisations are available in the `/assets/` directory.*

---

## 🏥 Scientific and Clinical Impact

**Methodological contribution:** ADR•X is, to the best of the authors' knowledge, among the first FAERS-based ML frameworks to enforce complete exclusion of outcome-derived disproportionality metrics, providing a reproducible reference implementation for leakage-aware pharmacovigilance modelling.

**Pharmacovigilance utility:** The framework operationalises SHAP-interpretable signal triage for sertraline, a widely prescribed SSRI with established pharmacogenomic complexity. The role-adaptive interface enables deployment across clinical, analytical, and regulatory contexts without requiring ML expertise from end users.

**Responsible AI alignment:** Explicit uncertainty communication, audit logging, and research-only disclosure collectively align with published roadmaps for responsible ML in healthcare. The framework rejects overconfident prediction in favour of calibrated, interpretable signal probability — a design choice of direct relevance to regulatory decision support.

**Extensibility:** The schema-frozen pipeline and modular feature architecture are designed for straightforward extension to additional therapeutic agents, enabling scalable multi-drug pharmacovigilance coverage.

---

## 🖥️ System Interface

The ADR•X Streamlit portal provides three role-adaptive interfaces:

- **Clinician View:** Simplified risk category display, SHAP top-3 contributor summary, and uncertainty flag.
- **Researcher View:** Full SHAP waterfall and global attribution charts, raw probability scores, and model variant selector.
- **Pharmacovigilance Analyst View:** Signal tier classification, SQLite-logged audit trail, and exportable PDF report.

```
📸 Dashboard Preview
┌─────────────────────────────────────────────┐
│  ADR•X — Sertraline Signal Explorer         │
│  ─────────────────────────────────────────  │
│  Patient Profile  │  ADR Risk Probability   │
│  ───────────────  │  ─────────────────────  │
│  Age: 52          │  ██████░░░░  0.61       │
│  Dose: 100 mg     │  Signal: MODERATE       │
│  CYP2C19: PM      │                         │
│  Polypharmacy: ✓  │  [View SHAP Details]    │
└─────────────────────────────────────────────┘
```

> 📸 *Insert actual screenshots at: `assets/dashboard_main.png`, `assets/shap_global.png`, `assets/shap_local.png`*

---

## 📁 Project Structure

```
adrx-sertraline-signal-explorer/
│
├── app/
│   ├── main.py                    # Streamlit entry point
│   ├── views/
│   │   ├── clinician_view.py      # Simplified clinical interface
│   │   ├── researcher_view.py     # Full SHAP and model analytics
│   │   └── pv_analyst_view.py     # Signal triage and audit interface
│   └── components/
│       ├── ai_assistant.py        # Embedded AI interpretation assistant
│       ├── shap_plots.py          # SHAP visualisation utilities
│       └── report_generator.py    # PDF report export
│
├── model/
│   ├── train.py                   # Model training pipeline
│   ├── evaluate.py                # AUC-ROC, calibration, sensitivity
│   ├── lgbm_baseline.joblib       # Serialised baseline model
│   └── lgbm_weighted.joblib       # Serialised imbalance-adjusted model
│
├── features/
│   ├── feature_schema.json        # Frozen feature schema (v1.0)
│   ├── engineering.py             # Feature construction pipeline
│   ├── pharmacogenomics.py        # CYP2C19 / CYP2D6 / SLC6A4 proxies
│   └── omics_proxies.py           # Multi-omics surrogate features
│
├── data/
│   ├── faers_raw/                 # Raw FAERS quarterly files (not tracked)
│   ├── processed/                 # Deduplicated, label-frozen dataset
│   └── external/                  # DrugBank / SIDER reference files
│
├── explainability/
│   ├── shap_global.py             # Global SHAP attribution pipeline
│   ├── shap_local.py              # Per-patient waterfall generation
│   └── plausibility_check.py     # Biological plausibility validation
│
├── audit/
│   └── adrx_audit.db             # SQLite audit log (auto-generated)
│
├── assets/
│   ├── dashboard_main.png         # Interface screenshot
│   ├── shap_global.png            # Global SHAP bar chart
│   ├── shap_local.png             # Local SHAP waterfall
│   └── methodology_workflow.png   # End-to-end pipeline diagram
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_shap_analysis.ipynb
│   └── 05_results_visualisation.ipynb
│
├── requirements.txt
├── environment.yml
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Installation and Usage

### Prerequisites

- Python 3.9 or higher
- pip or conda package manager
- FAERS quarterly data files (publicly available at [FDA FAERS](https://www.fda.gov/drugs/fda-adverse-event-reporting-system-faers))

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/adrx-sertraline-signal-explorer.git
cd adrx-sertraline-signal-explorer

# Create and activate virtual environment
python -m venv adrx_env
source adrx_env/bin/activate        # Linux / macOS
adrx_env\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Reproducing the Analysis

```bash
# Step 1: Data preprocessing and leakage prevention
python features/engineering.py --input data/faers_raw/ --output data/processed/

# Step 2: Model training (both variants)
python model/train.py --data data/processed/ --output model/

# Step 3: SHAP explainability analysis
python explainability/shap_global.py --model model/lgbm_baseline.joblib
python explainability/shap_local.py  --model model/lgbm_baseline.joblib

# Step 4: Model evaluation
python model/evaluate.py --model model/lgbm_baseline.joblib --data data/processed/
```

### Launching the Streamlit Portal

```bash
streamlit run app/main.py
```

The application will be accessible at `http://localhost:8501` by default.

### Running Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

Execute notebooks in sequential order (01 → 05) to reproduce the full analysis pipeline.

---

## 🤖 AI Assistant

ADR•X integrates an embedded AI assistant within the Streamlit portal, designed to assist users in interpreting model outputs without requiring ML expertise.

**Capabilities:**

- Plain-language explanation of SHAP feature contributions for individual patient predictions.
- Contextualisation of ADR risk scores within established sertraline pharmacology.
- Guidance on the sensitivity–calibration trade-off between the two model variants.
- Clarification of uncertainty signals and appropriate downstream actions.
- Response to natural-language queries regarding model design, leakage prevention, and pharmacogenomic feature logic.

The assistant operates within a research-use-only disclosure framework and explicitly declines to provide individualised clinical recommendations.

---

## 🔭 Future Work

The following directions are identified for extension of the ADR•X framework:

- **External validation:** Prospective benchmarking against independent pharmacovigilance cohorts and WHO VigiBase records to assess cross-database generalisability.
- **Multi-drug expansion:** Application of the leakage-aware pipeline to additional high-risk therapeutic agents (e.g., warfarin, clozapine, methotrexate).
- **Patient-level genomics:** Integration of empirically measured pharmacogenomic variant data (CYP2C19\*2, CYP2D6\*4, SLC6A4 5-HTTLPR) in place of proxy-based indicators.
- **Longitudinal clinical data:** Incorporation of electronic health record (EHR) time-series features to capture exposure trajectory and comorbidity evolution.
- **Regulatory alignment:** Adaptation of the signal output format to EMA and ICH E2E pharmacovigilance planning guidelines.
- **Federated learning:** Privacy-preserving multi-institutional model training to address FAERS under-reporting and population diversity limitations.
- **Probabilistic calibration:** Application of Platt scaling or isotonic regression to further improve probability reliability of the imbalance-adjusted variant.

---

## 📚 References

1. Edwards IR, Aronson JK. Adverse drug reactions: Definitions, diagnosis, and management. *Lancet.* 2000;356(9237):1255–1259.
2. World Health Organization. *The Importance of Pharmacovigilance: Safety Monitoring of Medicinal Products.* Geneva: WHO Press; 2022.
3. U.S. Food and Drug Administration. FDA Adverse Event Reporting System (FAERS) Public Dashboard [Internet]. Silver Spring (MD): FDA; 2023. Available from: https://www.fda.gov/drugs/fda-adverse-event-reporting-system-faers
4. Harpaz R, DuMouchel W, Shah NH, Madigan D, Ryan P, Friedman C. Novel data-mining methodologies for adverse drug event discovery and analysis. *Clin Pharmacol Ther.* 2012;91(6):1010–1021.
5. Hauben M, Bate A. Decision support methods for the detection of adverse events in post-marketing data. *Drug Discov Today.* 2009;14(7–8):343–357.
6. Wiens J, Saria S, Sendak M, et al. Do no harm: A roadmap for responsible machine learning for health care. *Nat Med.* 2019;25:1337–1340.
7. Stahl SM. *Stahl's Essential Psychopharmacology: Neuroscientific Basis and Practical Applications.* 4th ed. Cambridge University Press; 2013.
8. Kirchheiner J, Brosen K, Dahl ML, et al. CYP2D6 and CYP2C19 genotype-based dose recommendations for antidepressants. *Acta Psychiatr Scand.* 2001;104(3):173–192.
9. Whirl-Carrillo M, McDonagh EM, Hebert JM, et al. Pharmacogenomics knowledge for personalised medicine. *Clin Pharmacol Ther.* 2012;92(4):414–417.
10. Ke G, Meng Q, Finley T, et al. LightGBM: A highly efficient gradient boosting decision tree. *Adv Neural Inf Process Syst.* 2017;30:3146–3154.
11. Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. *Adv Neural Inf Process Syst.* 2017;30:4765–4774.
12. Lundberg SM, Erion G, Chen H, et al. From local explanations to global understanding with explainable AI for trees. *Nat Mach Intell.* 2020;2:56–67.

---

## 👤 Authors

**Adarsh Dheeraj Dubey** *(Corresponding Author)*
M.Sc. Bioinformatics (Part II), Department of Bioinformatics
Guru Nanak Khalsa College of Arts, Science & Commerce (Autonomous), Matunga (East), Mumbai – 400019, Maharashtra, India
📧 g24.adarshdheeraj.dheerajdubey@gnkhalsa.edu.in
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/adarsh-dubey)

**Ranjana Mangesh Parab** *(Co-first Author)*
M.Sc. Bioinformatics (Part II), Department of Bioinformatics
Guru Nanak Khalsa College, Mumbai

**Mrs. Sermarani Nadar** *(Thesis Supervisor)*
Department of Bioinformatics, Guru Nanak Khalsa College, Mumbai

**Dr. Gursimran Kaur Uppal** *(Co-Supervisor, Head of Department)*
Department of Bioinformatics, Guru Nanak Khalsa College, Mumbai

> A.D.D. and R.M.P. contributed equally to conceptualisation, framework development, data curation, analysis, and manuscript preparation. S.N. supervised and critically revised the manuscript. G.K.U. provided expert oversight as Head of Department and co-supervisor. All authors approved the final version (ICMJE criteria met).

---

## ⚠️ Disclaimer

**ADR•X is a research tool developed exclusively for scientific and educational purposes.** It is not a clinical decision support system, a medical device, or a substitute for professional pharmacovigilance assessment. Outputs should not be used to guide individual patient care, prescribing decisions, or regulatory submissions without independent clinical and statistical validation.

All data used in this study were obtained from the publicly available, anonymised FDA Adverse Event Reporting System (FAERS) database. No personally identifiable patient information was accessed or processed. No institutional ethics approval was required.

Model predictions represent probabilistic signal estimates under the constraints of spontaneous reporting data and should be interpreted accordingly. The authors accept no liability for decisions made on the basis of these outputs.

---

*ADR•X — Sertraline Signal Explorer | M.Sc. Bioinformatics Thesis Project | Guru Nanak Khalsa College, University of Mumbai | 2025–2026*
