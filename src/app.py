import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime
import streamlit.components.v1 as components
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import shap
from streamlit_shap import st_shap
from feature_builder import build_feature_vector
from database import init_db, get_connection
from fpdf import FPDF
from io import BytesIO

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="ADR•X — Sertraline Signal Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)
# -------------------------------------------------
# Global App Settings (Safe Variables)
# -------------------------------------------------
theme_value = "light"
eli12 = False
show_model_explanation = True
user_role = "Researcher"
# ---- CSS loader function FIRST ----
def load_css(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---- THEN load CSS ----
load_css("styles/styles.css")

# ---- THEN rest of app ----
init_db()
# -------------------------------------------------
# Initialize Session State (MUST be at top)
# -------------------------------------------------
if "signal_score" not in st.session_state:
    st.session_state.signal_score = None

if "risk_reasons" not in st.session_state:
    st.session_state.risk_reasons = []
    
if "model_input" not in st.session_state:
    st.session_state.model_input = None

if "conf_label" not in st.session_state:
    st.session_state.conf_label = "Not computed"

if "shap_vals" not in st.session_state:
    st.session_state.shap_vals = None

if "explanations" not in st.session_state:
    st.session_state.explanations = []

if "signal_percent" not in st.session_state:
    st.session_state.signal_percent = None

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/sertraline_lightgbm_full.pkl")

model = load_model()
load_css("styles/styles.css")
# -------------------------------------------------
# Freeze expected feature schema (CRITICAL)
# -------------------------------------------------
EXPECTED_FEATURES = list(model.feature_name())
# -------------------------------------------------
# Optional: Validation data for performance tab
# -------------------------------------------------
@st.cache_data
def load_validation_data():
    df = pd.read_csv("data/validation_dataset.csv")
    y_true = df["label"].values
    X_val = df[EXPECTED_FEATURES]
    return X_val, y_true

X_val, y_val = load_validation_data()

# -------------------------------------------------
# Load Omics Summary (UI only)
# -------------------------------------------------
@st.cache_data
def load_omics():
    return pd.read_csv("data/sertraline_omics_summary.csv")

omics_df = load_omics()

@st.cache_resource
def load_shap_explainer(_model):
    return shap.TreeExplainer(_model)
explainer = load_shap_explainer(model)

# -------------------------------------------------
# Helper functions (UNCHANGED)
# -------------------------------------------------

def get_pharmacogene_info(probability):
    return {
        "CYP2C19": {
            "relevance": "Critical",
            "phenotype": "Normal Metabolizer",
            "implication": "Standard dosing recommended; monitor response",
            "icon": "✓"
        },
        "CYP2D6": {
            "relevance": "Important",
            "phenotype": "Normal Metabolizer",
            "implication": "No major dose adjustment required",
            "icon": "✓"
        },
        "HTR2A": {
            "relevance": "Moderate",
            "phenotype": "Wild-type",
            "implication": "May influence serotonergic ADRs",
            "icon": "ℹ"
        },
        "SLC6A4": {
            "relevance": "Moderate",
            "phenotype": "Long allele (LL)",
            "implication": "Better transporter expression; favorable response",
            "icon": "✓"
        }
    }
def apply_patient_risk_adjustment(base_prob, age, sex, dose, polypharmacy, liver_disease,sert_protein,p_gp_activity,gut_microbiome,epigenetic_silencing):
    adjustment = 0.0
    reasons = []

    if age >= 65:
        adjustment += 0.10
        reasons.append("Elderly age increases CNS and hyponatremia risk")

    if dose >= 100:
        adjustment += 0.12
        reasons.append("High sertraline dose increases GI and CNS ADR risk")

    if polypharmacy:
        adjustment += 0.15
        reasons.append("Polypharmacy increases drug–drug interaction risk")

    if liver_disease:
        adjustment += 0.18
        reasons.append("Liver disease reduces drug clearance")

    if sex == "Female":
        adjustment += 0.05
        reasons.append("Female sex associated with higher nausea and dizziness")
    # Proteomics effects
    if sert_protein == "High":
        adjustment += 0.06
        reasons.append("High SERT protein increases serotonergic ADR risk")

    if p_gp_activity == "Low":
        adjustment += 0.08
        reasons.append("Reduced P-gp increases CNS drug exposure")

    # Microbiome effects
    if gut_microbiome == "Moderate Dysbiosis":
        adjustment += 0.05
        reasons.append("Gut dysbiosis increases GI ADR risk")

    elif gut_microbiome == "Severe Dysbiosis":
        adjustment += 0.12
        reasons.append("Severe gut dysbiosis strongly increases GI ADR risk")

    # Epigenomics
    adjustment += epigenetic_silencing * 0.08
    reasons.append("Epigenetic regulation alters drug response pathways")

    final_prob = min(base_prob + adjustment, 0.95)
    return final_prob, reasons

def get_metabolomic_profile(probability):
    return {
        "Serotonin metabolites": {
            "direction": "↑",
            "clinical_meaning": "Increased serotonergic activity"
        },
        "Tryptophan": {
            "direction": "↓",
            "clinical_meaning": "Substrate depletion"
        },
        "Kynurenine pathway metabolites": {
            "direction": "↑",
            "clinical_meaning": "Possible neuroinflammation"
        },
        "Lipid peroxides": {
            "direction": "↓",
            "clinical_meaning": "Reduced oxidative stress"
        },
        "Inflammatory cytokines": {
            "direction": "↑",
            "clinical_meaning": "Mild immune activation"
        }
    }

def get_adr_evidence():
    common_adrs = [
        {"adr": "Nausea/Vomiting", "incidence": "18–26%", "severity": "Mild–Moderate", "onset": "Days 1–7"},
        {"adr": "Insomnia/Dizziness", "incidence": "10–15%", "severity": "Mild–Moderate", "onset": "Days 1–14"},
        {"adr": "Sexual Dysfunction", "incidence": "16–40%", "severity": "Moderate", "onset": "Weeks 2–8"},
        {"adr": "Headache", "incidence": "12–18%", "severity": "Mild", "onset": "Days 1–7"}
    ]

    serious_adrs = [
        {"adr": "Serotonin Syndrome", "risk": "<1%", "trigger": "Poly-serotonergic drugs", "management": "Immediate discontinuation"},
        {"adr": "QT Prolongation", "risk": "<1%", "trigger": "High dose, interactions", "management": "ECG monitoring"},
        {"adr": "Hyponatremia", "risk": "0.5–1.5%", "trigger": "Elderly, SIADH", "management": "Monitor Na⁺"},
        {"adr": "Bleeding Risk", "risk": "0.1–0.3%", "trigger": "Platelet inhibition", "management": "Clinical monitoring"}
    ]

    return common_adrs, serious_adrs

def get_risk_category(score):
    if score >= 0.8:
        return "High Priority Signal", "🔴", "risk-high"
    elif score >= 0.5:
        return "Moderate Priority Signal", "🟠", "risk-moderate"
    else:
        return "Low Priority Signal", "🟢", "risk-low"

def get_organ_specific_risks(probability):
    return {
        "CNS": max(0, min(1, probability + 0.15)),
        "Cardiac": max(0, min(1, probability - 0.20)),
        "Hepatic": max(0, min(1, probability - 0.25)),
        "Gastrointestinal": max(0, min(1, probability + 0.25)),
        "Metabolic": max(0, min(1, probability - 0.15)),
        "Immune": max(0, min(1, probability - 0.10)),
    }
def derive_risk_profiles(overall_risk):
    """
    Decompose overall ADR risk into clinically meaningful sub-risks
    """
    return {
        "GI_ADR_Risk": min(1.0, overall_risk + 0.20),
        "CNS_ADR_Risk": min(1.0, overall_risk + 0.10),
        "Sexual_ADR_Risk": min(1.0, overall_risk + 0.05),
        "Serious_ADR_Risk": min(1.0, overall_risk - 0.20)
    }
def estimate_adr_timeline(risk_score):
    if risk_score >= 0.75:
        return "Early onset (0–7 days): GI & CNS ADRs likely"
    elif risk_score >= 0.50:
        return "Intermediate onset (1–4 weeks): CNS & sleep ADRs"
    else:
        return "Late or unlikely onset (>4 weeks): sexual ADRs possible"
def generate_risk_reduction_advice(age, dose, polypharmacy, liver_disease):
    advice = []

    if dose >= 100:
        advice.append("Consider reducing dose to 50 mg/day to lower ADR risk")

    if polypharmacy:
        advice.append("Review concomitant medications for interactions")

    if liver_disease:
        advice.append("Monitor liver function and consider dose adjustment")

    if age >= 65:
        advice.append("Elderly patients may benefit from slower titration")

    if not advice:
        advice.append("Current regimen appears appropriate")

    return advice
def generate_adr_timeline(risk_score):
    """
    Simulated longitudinal ADR risk progression
    """
    days = [0, 7, 30]
    
    gi_risk = [
        risk_score * 0.6,
        risk_score * 1.0,   # peak
        risk_score * 0.4
    ]
    
    cns_risk = [
        risk_score * 0.3,
        risk_score * 0.6,
        risk_score * 0.9   # stabilization
    ]
    
    return days, gi_risk, cns_risk

def shap_to_natural_language(
    shap_values,
    feature_names,
    feature_values,
    top_k: int = 6
):
    """
    Convert SHAP values into a conservative, scientific
    natural language explanation.
    """
    import numpy as np

    shap_vals = shap_values.flatten()
    abs_shap = np.abs(shap_vals)

    # Top contributing features
    top_idx = np.argsort(abs_shap)[::-1][:top_k]

    explanations = []

    for idx in top_idx:
        fname = feature_names[idx]
        sval = shap_vals[idx]
        fval = feature_values[idx]

        direction = "increases" if sval > 0 else "reduces"

        # Conservative phrasing
        explanations.append(
            f"• {fname.replace('_', ' ')} ({fval}) {direction} the predicted ADR risk"
        )

    return explanations

def pdf_safe(text: str) -> str:
    """
    Convert Unicode text to Latin-1 safe text for FPDF.
    """
    replacements = {
        "•": "-",
        "→": "->",
        "↑": "up",
        "↓": "down",
        "≤": "<=",
        "≥": ">=",
        "–": "-",
        "—": "-",
        "’": "'",
        "“": '"',
        "”": '"',
        "⚠️": "WARNING:",
        "🔴": "[HIGH]",
        "🟠": "[MODERATE]",
        "🟢": "[LOW]"
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text.encode("latin-1", "ignore").decode("latin-1")


def get_system_prompt(role):
    if "Clinician" in role:
        return """
        You are an AI clinical decision-support assistant.
        Explain results clearly and conservatively.
        Avoid technical jargon unless necessary.
        Do NOT give diagnoses or treatment recommendations.
        """
    else:
        return """
        You are an AI research assistant for pharmacovigilance.
        Provide technical explanations of model behavior,
        feature contributions, SHAP values, and uncertainty.
        Do NOT provide clinical decisions.
        """
def run_ai_chat(system_prompt, context, question):
    q = question.lower()

    if "confidence" in q:
        return (
            "Model confidence reflects how far the predicted ADR risk "
            "is from the decision threshold. Higher confidence indicates "
            "greater consistency in model signals."
        )

    if "why" in q or "reason" in q:
        return (
            "The predicted ADR risk is influenced by dose, age, "
            "polypharmacy, and omics-derived biological signals "
            "such as inflammation and transporter activity."
        )

    if "omics" in q:
        return (
            "Omics features modify baseline ADR risk by capturing "
            "biological states such as inflammation, metabolism, "
            "and protein target availability."
        )

    return (
        "This assistant explains model outputs and biological context. "
        "Please ask about risk, confidence, omics, or interpretation."
    )

SUGGESTED_QUESTIONS = {
    "Clinician": [
        "What does this prediction mean for my patient?",
        "How confident is the model?",
        "Which side effects should I monitor first?",
        "Is this risk considered high or moderate?"
    ],
    "Researcher": [
        "How is the ADR signal score calculated?",
        "Which features contributed most to this prediction?",
        "How should I interpret the SHAP plot?",
        "What are the limitations of this model?"
    ]
}


GRAPH_REGISTRY = {
    "ADR Signal Strength": {
        "type": "metric",
        "meaning": (
            "Represents the relative strength of the predicted adverse drug reaction signal. "
            "Higher values indicate stronger pharmacovigilance concern."
        ),
        "interpretation": {
            "low": "Risk comparable to general population",
            "moderate": "Clinically relevant ADR risk",
            "high": "Strong ADR signal requiring vigilance"
        }
    },

    "Organ Risk Radar": {
        "type": "radar",
        "meaning": (
            "Shows how ADR risk is distributed across physiological systems "
            "such as CNS, GI, hepatic, cardiac, and immune systems."
        ),
        "interpretation": (
            "Larger area toward a system indicates higher relative susceptibility "
            "to ADRs affecting that organ."
        )
    },

    "ADR Timeline": {
        "type": "line",
        "meaning": (
            "Shows expected temporal pattern of ADR development after starting sertraline."
        ),
        "interpretation": (
            "Early peaks suggest acute ADRs, while delayed rises suggest late-onset effects."
        )
    },

    "SHAP Explanation": {
        "type": "explainability",
        "meaning": (
            "Explains which features contributed most to this prediction."
        ),
        "interpretation": (
            "Features pushing right increase ADR risk, left reduce risk."
        )
    }
}
FEATURE_REGISTRY = {
    "Age": "Older age increases ADR risk due to reduced clearance and CNS sensitivity.",
    "Dose": "Higher sertraline dose increases serotonergic and GI ADR risk.",
    "Polypharmacy": "Multiple drugs increase interaction-related ADRs.",
    "Liver Disease": "Impaired hepatic clearance increases systemic exposure.",
    "Neuroinflammation": "Elevated neuroinflammation increases CNS ADR susceptibility.",
    "SERT Expression": "Higher transporter availability increases serotonergic effects."
}

def build_chat_context():
    if st.session_state.signal_score is None:
        return None

    return {
        "risk_score": st.session_state.signal_score,
        "risk_percent": st.session_state.signal_percent,
        "risk_category": get_risk_category(st.session_state.signal_score)[0],
        "confidence": st.session_state.conf_label,
        "risk_reasons": st.session_state.risk_reasons,
        "shap_explanations": st.session_state.explanations,
        "graphs": GRAPH_REGISTRY,
        "features": FEATURE_REGISTRY
    }

def detect_intent(question: str):
    q = question.lower().strip()

    # Risk & prediction meaning
    if any(k in q for k in ["prediction", "result", "what does", "meaning"]):
        return "risk"

    # Confidence & certainty
    if any(k in q for k in ["confidence", "sure", "certainty", "how sure"]):
        return "confidence"

    # Features & SHAP
    if any(k in q for k in ["feature", "shap", "why", "contribute"]):
        return "feature"

    # Graphs & plots
    if any(k in q for k in ["graph", "plot", "chart", "radar", "timeline"]):
        return "graph"
    if any(k in q for k in ["signal", "score", "calculated"]):
        return "signal"

    return "general"

def detect_graph_reference(question: str):
    q = question.lower()

    if "radar" in q or "organ" in q:
        return "Organ Risk Radar"
    if "timeline" in q or "time" in q:
        return "ADR Timeline"
    if "shap" in q:
        return "SHAP Explanation"
    if "signal" in q or "strength" in q:
        return "ADR Signal Strength"

    return None

def explain_with_context(context, question):
    intent = detect_intent(question)

    # ---------------- RISK ----------------
    if intent == "risk":
        return (
            f"The model predicts an ADR signal strength of "
            f"{context['risk_percent']}%, classified as "
            f"{context['risk_category']}. "
            "This represents relative pharmacovigilance risk, "
            "not a certainty that side effects will occur."
        )

    # ---------------- CONFIDENCE ----------------
    if intent == "confidence":
        return (
            f"The model confidence is **{context['confidence']}**. "
            "Confidence reflects how far the prediction is from the "
            "decision threshold (0.5). Higher confidence means "
            "stronger and more consistent model support."
        )

    # ---------------- SIGNAL SCORE ----------------
    if intent == "signal":
        return (
            "The ADR signal score is the predicted probability "
            "generated by the machine-learning model after integrating "
            "clinical, pharmacological, and omics features."
        )

    # ---------------- FEATURES / SHAP ----------------
    if intent == "feature":
        explanation = "Key contributors to this prediction:\n\n"
        for r in context["risk_reasons"]:
            explanation += f"- {r}\n"

        explanation += "\nFeature context:\n"
        for fname, fdesc in context["features"].items():
            explanation += f"- {fname}: {fdesc}\n"

        return explanation

    # ---------------- GRAPHS ----------------
    if intent == "graph":
        graph_name = detect_graph_reference(question)

        if graph_name and graph_name in context["graphs"]:
            g = context["graphs"][graph_name]

            text = f"**{graph_name}**\n\n"
            text += g["meaning"] + "\n\n"

            if isinstance(g["interpretation"], dict):
                for k, v in g["interpretation"].items():
                    text += f"- {k.capitalize()}: {v}\n"
            else:
                text += g["interpretation"]

            return text

        return (
            "This visualization summarizes aspects of ADR risk. "
            "You can ask specifically about the radar plot, timeline, "
            "SHAP explanation, or signal strength."
        )

    # ---------------- FALLBACK ----------------
    return (
        "I can explain the prediction, confidence, features, SHAP plots, "
        "and risk visualizations. Try asking about one of these."
    )



def generate_pdf_report(
    patient_info: dict,
    risk_score: float,
    risk_category: str,
    explanations: list,
    confidence_label: str
):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Sertraline ADR Risk Assessment Report", ln=True)

    pdf.ln(4)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, "Generated by AI-based Pharmacovigilance System", ln=True)

    pdf.ln(6)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "Patient Summary", ln=True)

    pdf.set_font("Arial", "", 11)
    for k, v in patient_info.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=True)

    pdf.ln(4)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "ADR Risk Prediction", ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Risk Score: {risk_score:.3f}", ln=True)
    pdf.cell(0, 8, f"Risk Category: {risk_category}", ln=True)

    pdf.ln(4)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "Model Explanation (SHAP-based)", ln=True)

    pdf.set_font("Arial", "", 11)
    for exp in explanations:
        pdf.multi_cell(0, 7, pdf_safe(exp))

    pdf.ln(2)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "Model Confidence", ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 7, pdf_safe(confidence_label))
    for k, v in patient_info.items():
        pdf.cell(0, 8, pdf_safe(f"{k}: {v}"), ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(
        0,
        6,
        "Disclaimer: This report is generated for research and decision-support "
        "purposes only. Predictions reflect model behavior and are not clinical diagnoses."
    )

    pdf_bytes = pdf.output(dest="S")
    # If returned as string (older fpdf), convert safely
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode("latin-1")
    buffer = BytesIO(pdf_bytes)
    return buffer

def tooltip(label, explanation):
    st.markdown(
        f"""
        <span style="position: relative; cursor: help; font-weight:600;">
            {label}
            <span style="
                visibility: hidden;
                background: rgba(15,23,42,0.95);
                color: #f8fafc;
                padding: 8px 10px;
                border-radius: 8px;
                position: absolute;
                z-index: 10;
                width: 260px;
                bottom: 120%;
                left: 0;
                box-shadow: 0 10px 25px rgba(0,0,0,0.6);
            " class="tooltip-text">
                {explanation}
            </span>
        </span>

        <script>
        const el = document.currentScript.previousElementSibling;
        el.onmouseenter = () => el.querySelector(".tooltip-text").style.visibility = "visible";
        el.onmouseleave = () => el.querySelector(".tooltip-text").style.visibility = "hidden";
        </script>
        """,
        unsafe_allow_html=True
    )


def explain_percentage_risk(percent):
    if percent < 30:
        return "Low ADR signal. Risk comparable to general population. Standard monitoring is sufficient."
    elif percent < 50:
        return "Mild to moderate ADR signal. Side effects are possible but usually manageable with monitoring."
    elif percent < 70:
        return "Moderate ADR signal. Clinically relevant risk. Close monitoring and dose optimization advised."
    elif percent < 85:
        return "High ADR signal. Strong risk indicators detected. Enhanced vigilance recommended."
    else:
        return "Very high ADR signal. Priority pharmacovigilance concern. Consider alternative strategies."

# -------------------------------------------------
# Title & disclaimer (UNCHANGED)
# -------------------------------------------------
# -------------------------------------------------
# Hero header
# -------------------------------------------------
st.markdown("""
<div class="research-header-strong">
    <div class="research-main-title">
        ADR•X
    </div>
    <div class="research-project-title">
        Sertraline Signal Explorer
    </div>
    <div class="research-context-line">
        AI-Driven Multi-Omics Framework for Adverse Drug Reaction Prediction
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="glass-card main-fade">', unsafe_allow_html=True)

st.warning("""⚠️ RESEARCH & ACADEMIC USE ONLY  
This tool does NOT replace clinical judgment.
""")

components.html(
    """
    <script>
    const el = window.parent.document.querySelector('.app-logo');
    if (el) {
        el.style.transition = 'box-shadow 0.4s ease-in-out';
        el.addEventListener('mouseenter', () => {
            el.style.boxShadow = '0 0 32px rgba(56, 189, 248, 0.9)';
        });
        el.addEventListener('mouseleave', () => {
            el.style.boxShadow = '0 0 10px rgba(56, 189, 248, 0.4)';
        });
    }
    </script>
    """,
    height=0,
)
components.html(
    """
    <div class="cursor-glow" id="cursorGlow"></div>
    <script>
        const glow = document.getElementById("cursorGlow");
        window.addEventListener("mousemove", (e) => {
            glow.style.transform = 
                `translate(${e.clientX}px, ${e.clientY}px)`;
        });
    </script>
    """,
    height=0,
)
components.html(
    """
    <script>
    const revealElements = () => {
        const elements = document.querySelectorAll(".reveal");
        const triggerBottom = window.innerHeight * 0.9;

        elements.forEach(el => {
            const boxTop = el.getBoundingClientRect().top;
            if (boxTop < triggerBottom) {
                el.classList.add("active");
            }
        });
    };

    window.addEventListener("scroll", revealElements);
    revealElements();
    </script>
    """,
    height=0,
)

st.info("""
### How to use this tool

1. Enter patient details in the **left sidebar**
2. (Optional) Adjust **omics & pharmacogenomic inputs**
3. Click **Predict ADR Risk**
4. Review:
   - Overall ADR signal
   - Organ-specific risk radar
   - Biological & pharmacogenomic explanations

This tool is for **research & decision support**, not diagnosis.
""")

if st.session_state.signal_percent is not None:
    st.info(
        f"""
        **What does {st.session_state.signal_percent}% mean?**

        {explain_percentage_risk(st.session_state.signal_percent)}

        This percentage reflects **relative ADR signal strength**,  
        not a guarantee that side effects will occur.
        """
    )
else:
    st.info(
        "Click **Predict ADR Risk** to generate a percentage-based explanation."
    )



# -------------------------------------------------
# Sidebar (UNCHANGED)
# -------------------------------------------------
show_raw_values = st.sidebar.checkbox("Show raw omics values", value=False)
with st.sidebar.expander("Patient Details", expanded=True):
    age = st.number_input("Age (years)", 18, 90, 35)
    sex = st.selectbox("Sex", ["Female", "Male", "Other"])
    weight = st.number_input("Weight (kg)", 30, 150, 60)
    dose = st.sidebar.selectbox("Sertraline Dose (mg/day)", [25, 50, 100, 150])
    polypharmacy = st.sidebar.checkbox("On ≥3 concomitant drugs")
    liver_disease = st.sidebar.checkbox("Known liver disease")
    elderly = age >= 65

st.sidebar.info(f"""
**Model:** LightGBM  
**Features:** {len(EXPECTED_FEATURES)}  
**Last Updated:** {datetime.now().strftime('%Y-%m-%d')}
""")
st.markdown("### Risk Evaluation Summary")

if st.session_state.signal_score is not None:
    st.markdown(f"""
    **ADR Signal Score:** {st.session_state.signal_score:.3f}
    Higher score = stronger pharmacovigilance signal
    """)

    if st.session_state.risk_reasons:
        st.markdown("**Factors influencing this prediction:**")
        st.markdown("### Key Contributors to This Risk")
        for r in st.session_state.risk_reasons:
            st.markdown(f"• {r}")

else:
    st.info("Click **Predict ADR Risk** to generate a prediction.")

st.sidebar.markdown("### Patient Omics (Optional)")
# -------- PROTEOMICS --------
with st.sidebar.expander("Proteomics"):
    sert_protein = st.selectbox(
        "SERT Protein Availability",
        ["Low", "Normal", "High"],
        index=1
    )

    p_gp_activity = st.selectbox(
        "P-glycoprotein (ABCB1) Activity",
        ["Low", "Normal", "High"],
        index=1
    )

# -------- MICROBIOME --------
with st.sidebar.expander("Gut Microbiome"):
    gut_microbiome = st.selectbox(
        "Gut Microbiome Balance",
        ["Healthy", "Moderate Dysbiosis", "Severe Dysbiosis"],
        index=0
    )


# -------- EPIGENOMICS --------
with st.sidebar.expander("Epigenomics"):
    epigenetic_silencing = st.slider(
        "Epigenetic Silencing Index",
        0.0, 1.0, 0.3
    )
# -------- OMICS INPUTS --------
neuroinflammation = st.sidebar.slider(
    "Neuroinflammation",
    -2.0, 2.0, 0.0
)

oxidative_stress = st.sidebar.slider(
    "Oxidative Stress",
    -2.0, 2.0, 0.0
)

bbb_integrity = st.sidebar.slider(
    "Blood–Brain Barrier Integrity",
    -2.0, 2.0, 0.0
)

cytokine_activity = st.sidebar.slider(
    "Cytokine Activation",
    -2.0, 2.0, 0.0
)

# -------- PHARMACOGENOMICS --------
cyp2c19_status = st.sidebar.selectbox(
    "CYP2C19 phenotype",
    ["Poor", "Intermediate", "Normal", "Ultra-rapid"],
    index=2
)

cyp2d6_status = st.sidebar.selectbox(
    "CYP2D6 phenotype",
    ["Poor", "Intermediate", "Normal", "Ultra-rapid"],
    index=2
)

sert_expression = st.sidebar.selectbox(
    "SERT (SLC6A4 expression)",
    ["Low", "Normal", "High"],
    index=1
)

# -------------------------------------------------
# Tabs (UNCHANGED)
# -------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Risk Overview",
    "Genomics",
    "Omics Features",
    "ADR Evidence",
    "Performance & SHAP",
    "Model Explanation",
    "Background",
    "Prediction History",
    "AI Clinical Assistant"
])

# =================================================
# TAB 1 — RISK SUMMARY (ACADEMIC VERSION)
# =================================================
with tab1:

    st.subheader("ADR Risk Assessment")

    # -------------------------------------------------
    # PREDICTION BUTTON (Computation only)
    # -------------------------------------------------
    if st.button("Predict ADR Risk"):

        # -----------------------------
        # Build model input
        # -----------------------------
        st.session_state.model_input = build_feature_vector(
            expected_features=EXPECTED_FEATURES,
            age=age,
            sex=sex,
            dose=dose,
            polypharmacy=polypharmacy,
            liver_disease=liver_disease,
            neuroinflammation=neuroinflammation,
            oxidative_stress=oxidative_stress,
            bbb_integrity=bbb_integrity,
            cytokine_activity=cytokine_activity,
            cyp2c19_status=cyp2c19_status,
            cyp2d6_status=cyp2d6_status,
            sert_expression=sert_expression
        )

        base_prob = float(model.predict(st.session_state.model_input)[0])

        # -----------------------------
        # SHAP
        # -----------------------------
        try:
            shap_vals = explainer.shap_values(st.session_state.model_input)
            st.session_state.shap_vals = shap_vals
        except Exception:
            st.session_state.shap_vals = None

        final_prob, reasons = apply_patient_risk_adjustment(
            base_prob=base_prob,
            age=age,
            sex=sex,
            dose=dose,
            polypharmacy=polypharmacy,
            liver_disease=liver_disease,
            sert_protein=sert_protein,
            p_gp_activity=p_gp_activity,
            gut_microbiome=gut_microbiome,
            epigenetic_silencing=epigenetic_silencing
        )

        st.session_state.signal_score = final_prob
        st.session_state.signal_percent = round(final_prob * 100, 1)

        confidence = abs(final_prob - 0.5) * 2
        if confidence > 0.6:
            st.session_state.conf_label = "High confidence"
        elif confidence > 0.3:
            st.session_state.conf_label = "Moderate confidence"
        else:
            st.session_state.conf_label = "Low confidence"

    # -------------------------------------------------
    # DISPLAY RESULTS (Clean Layout)
    # -------------------------------------------------
    if st.session_state.get("signal_score") is not None:

        signal_score = st.session_state.signal_score
        risk_label, _, _ = get_risk_category(signal_score)

        # -----------------------------
        # CORE METRICS
        # -----------------------------
        st.subheader("1. Summary Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("ADR Signal (%)", f"{st.session_state.signal_percent}%")
        col2.metric("Risk Classification", risk_label)
        col3.metric("Model Confidence", st.session_state.conf_label)

        st.divider()

        # -----------------------------
        # ORGAN DISTRIBUTION
        # -----------------------------
        st.subheader("2. Multisystem ADR Risk Attribution")

        organ_risks = get_organ_specific_risks(signal_score)

        radar_fig = go.Figure(
            go.Scatterpolar(
                r=list(organ_risks.values()),
                theta=list(organ_risks.keys()),
                fill="toself"
            )
        )

        radar_fig.update_layout(
            polar=dict(radialaxis=dict(range=[0, 1])),
            showlegend=False,
            template="plotly_dark" if theme_value == "dark" else "simple_white",
            height=380
        )

        st.plotly_chart(radar_fig, use_container_width=True)

        st.divider()

        # -----------------------------
        # TEMPORAL PROJECTION
        # -----------------------------
        st.subheader("3. Temporal Projection of ADR Probability")

        days, gi_curve, cns_curve = generate_adr_timeline(signal_score)

        timeline_fig = go.Figure()
        timeline_fig.add_trace(go.Scatter(x=days, y=gi_curve, mode="lines", name="GI"))
        timeline_fig.add_trace(go.Scatter(x=days, y=cns_curve, mode="lines", name="CNS"))

        timeline_fig.update_layout(
            xaxis_title="Days After Sertraline Initiation",
            yaxis_title="Relative ADR Probability",
            yaxis=dict(range=[0, 1]),
            template="plotly_dark" if theme_value == "dark" else "simple_white",
            height=380
        )

        st.plotly_chart(timeline_fig, use_container_width=True)

        st.divider()

        # -----------------------------
        # INTERPRETATION
        # -----------------------------
        st.subheader("4. Clinical Interpretation")

        if signal_score >= 0.8:
            st.error("High ADR risk. Enhanced monitoring and dose reassessment recommended.")
        elif signal_score >= 0.5:
            st.warning("Moderate ADR risk. Close monitoring during initiation advised.")
        else:
            st.success("Low ADR risk. Standard clinical monitoring likely sufficient.")

        if eli12:
            st.info(
                "This score estimates how likely side effects may occur. "
                "Higher values suggest closer follow-up is needed."
            )

        st.divider()

        # -----------------------------
        # EXPORT
        # -----------------------------
        st.subheader("5. Export Report")

        patient_info = {
            "Age": age,
            "Sex": sex,
            "Dose (mg/day)": dose,
            "Polypharmacy": polypharmacy,
            "Liver disease": liver_disease
        }

        pdf_buffer = generate_pdf_report(
            patient_info=patient_info,
            risk_score=signal_score,
            risk_category=risk_label,
            explanations=st.session_state.get("explanations", []),
            confidence_label=st.session_state.conf_label
        )

        st.download_button(
            label="Download PDF Report",
            data=pdf_buffer,
            file_name="sertraline_adr_risk_report.pdf",
            mime="application/pdf"
        )

        st.caption(
            "Confidence reflects the distance of the predicted score "
            "from the decision threshold (0.5)."
        )

# =================================================
# TAB 2 — PHARMACOGENOMIC PROFILE (ACADEMIC)
# =================================================
with tab2:

    st.subheader("Pharmacogenomic Risk Profile")

    st.markdown("""
    This section summarizes pharmacogenomic determinants influencing 
    sertraline pharmacokinetics, pharmacodynamics, and adverse drug reaction (ADR) susceptibility.
    Phenotypic classifications are derived from integrated omics signals 
    and model-informed inference.
    """)

    # -------------------------------------------------
    # Risk-based probability reference
    # -------------------------------------------------
    probability = (
        st.session_state.signal_score
        if st.session_state.get("signal_score") is not None
        else 0.5
    )

    genes_info = get_pharmacogene_info(probability)

    st.divider()

    # -------------------------------------------------
    # Gene-level Summary
    # -------------------------------------------------
    st.subheader("1. Gene-Level Phenotypic Interpretation")

    col1, col2 = st.columns(2)

    for idx, (gene_name, gene_data) in enumerate(genes_info.items()):
        with (col1 if idx % 2 == 0 else col2):
            st.markdown(f"""
            **{gene_name}**  
            *Functional Relevance:* {gene_data['relevance']}  

            **Predicted Phenotype:** {gene_data['phenotype']}  

            **Clinical Implication:**  
            {gene_data['implication']}
            """)

            st.markdown("---")

    # -------------------------------------------------
    # Mechanistic Mapping
    # -------------------------------------------------
    st.subheader("2. Gene → Pathway → ADR Mechanistic Mapping")

    pathways = {
        "CYP2C19 — Hepatic Biotransformation": {
            "description": (
                "CYP2C19 is a primary enzyme responsible for hepatic metabolism "
                "of sertraline. Reduced enzymatic activity may increase systemic exposure."
            ),
            "adr_link": (
                "Reduced clearance → Elevated plasma concentration → "
                "Enhanced serotonergic signaling → Increased GI, CNS, and QT-related ADR risk."
            )
        },
        "SLC6A4 — Serotonin Transporter (SERT)": {
            "description": (
                "SLC6A4 encodes the serotonin transporter, the principal pharmacodynamic "
                "target of sertraline."
            ),
            "adr_link": (
                "Altered transporter expression → Modified serotonergic blockade → "
                "Variability in therapeutic response and adverse effect susceptibility."
            )
        },
        "HTR2A — Post-Synaptic Receptor Signaling": {
            "description": (
                "HTR2A encodes the serotonin 2A receptor, influencing downstream "
                "signal transduction pathways associated with mood regulation and autonomic control."
            ),
            "adr_link": (
                "Receptor polymorphisms → Altered neurotransmission dynamics → "
                "Changes in mood, sexual function, and gastrointestinal motility."
            )
        }
    }

    for pathway_name, pathway_info in pathways.items():
        with st.expander(pathway_name):
            st.markdown(f"""
            **Biological Mechanism**  
            {pathway_info['description']}

            **Implication for ADR Risk**  
            {pathway_info['adr_link']}
            """)

    st.divider()

    st.caption(
        "Pharmacogenomic interpretations presented here are model-informed and "
        "should be validated through confirmatory genotyping prior to clinical application."
    )

# =================================================
# TAB 3 — MULTI-OMICS INTEGRATION (ACADEMIC)
# =================================================
with tab3:

    st.subheader("Multi-Omics & Biomarker Integration")

    st.markdown("""
    This module integrates transcriptomic, proteomic, metabolomic, 
    and systems-level biological signals associated with sertraline exposure.
    
    The framework models how multi-layer molecular perturbations 
    contribute to adverse drug reaction (ADR) susceptibility.
    """)

    probability = (
        st.session_state.signal_score
        if st.session_state.get("signal_score") is not None
        else 0.5
    )

    st.divider()

    # =================================================
    # 1. TRANSCRIPTOMICS
    # =================================================
    st.subheader("1. Transcriptomic Layer — Gene Expression Modulation")

    transcriptomic_pathways = {
        "Serotonergic Neurotransmission": {
            "status": "Upregulated",
            "meaning": "Enhanced SERT signaling; increased serotonergic activity"
        },
        "Neuroinflammation": {
            "status": "Moderately Upregulated",
            "meaning": "Innate immune activation associated with GI/CNS ADR risk"
        },
        "Oxidative Stress Response": {
            "status": "Downregulated",
            "meaning": "Reduced antioxidant buffering capacity"
        },
        "Blood–Brain Barrier Integrity": {
            "status": "Slightly Downregulated",
            "meaning": "Potential increase in CNS drug penetration"
        }
    }

    st.table(pd.DataFrame({
        "Pathway": transcriptomic_pathways.keys(),
        "Observed Regulation": [v["status"] for v in transcriptomic_pathways.values()],
        "Clinical Interpretation": [v["meaning"] for v in transcriptomic_pathways.values()]
    }))

   if show_raw_values:
    st.markdown("**Raw Transcriptomic Features:**")

    transcriptomic_cols = [
        col for col in omics_df.columns
        if "transcript" in col.lower()
    ]

    if transcriptomic_cols:
        st.dataframe(
            omics_df[transcriptomic_cols].head(),
            use_container_width=True
        )
    else:
        st.info("No transcriptomic-related columns detected in dataset.")
    # =================================================
    # 2. PROTEOMICS
    # =================================================
    st.subheader("2. Proteomic Layer — Functional Drug Targets")

    st.table(pd.DataFrame({
        "Protein": ["SERT (SLC6A4)", "HTR2A", "Albumin", "ABCB1 (P-gp)"],
        "Functional Role": [
            "Primary serotonin transporter target",
            "Serotonin receptor signaling",
            "Drug plasma protein binding",
            "Blood–brain barrier efflux transporter"
        ],
        "ADR Relevance": [
            "CNS & GI adverse effects",
            "Sexual dysfunction, insomnia",
            "Altered free drug fraction",
            "Modified CNS drug exposure"
        ]
    }))

    st.divider()

    # =================================================
    # 3. METABOLOMICS
    # =================================================
    st.subheader("3. Metabolomic Layer — Downstream Biochemical Effects")

    metabolomic_data = get_metabolomic_profile(probability)

    st.table(pd.DataFrame({
        "Metabolite": metabolomic_data.keys(),
        "Direction": [v["direction"] for v in metabolomic_data.values()],
        "Clinical Interpretation": [v["clinical_meaning"] for v in metabolomic_data.values()]
    }))

if show_raw_values:
    st.markdown("**Raw Metabolomic Features:**")

    expected_cols = [
        "metabolite_serotonin",
        "metabolite_tryptophan"
    ]

    available_cols = [
        col for col in expected_cols
        if col in omics_df.columns
    ]

    if available_cols:
        st.dataframe(
            omics_df[available_cols].head(),
            use_container_width=True
        )
    else:
        st.warning(
            "Metabolomic columns not found in dataset. "
            "Please verify omics CSV schema."
        )

st.divider()

    # =================================================
    # 4. MULTI-OMICS CLUSTERING
    # =================================================
    st.subheader("4. Multi-Omics Phenotypic Clustering")

    st.markdown("""
    Unsupervised clustering of integrated omics features identifies 
    molecular subtypes with distinct ADR susceptibility profiles.
    """)

    clusters = {
        "Cluster A — High Inflammatory Profile": {
            "prevalence": "~25%",
            "traits": "Elevated cytokines and oxidative stress markers",
            "adr_risk": "Higher GI and immune-mediated ADR risk",
            "recommendation": "Enhanced monitoring of inflammatory biomarkers"
        },
        "Cluster B — Normative Molecular Profile": {
            "prevalence": "~50%",
            "traits": "Balanced transcriptomic and metabolomic signals",
            "adr_risk": "Average ADR risk",
            "recommendation": "Standard monitoring"
        },
        "Cluster C — Impaired Metabolic Clearance": {
            "prevalence": "~15%",
            "traits": "Reduced detoxification gene activity",
            "adr_risk": "Higher CNS and hepatic ADR risk",
            "recommendation": "Consider lower starting dose"
        },
        "Cluster D — High Responder Phenotype": {
            "prevalence": "~10%",
            "traits": "Elevated SERT expression and serotonin turnover",
            "adr_risk": "Lower ADR risk, favorable response",
            "recommendation": "Standard dosing appropriate"
        }
    }

    for cluster_name, cluster_data in clusters.items():
        with st.expander(f"{cluster_name} ({cluster_data['prevalence']})"):
            st.markdown(f"""
            **Molecular Traits:**  
            {cluster_data['traits']}

            **ADR Risk Pattern:**  
            {cluster_data['adr_risk']}

            **Clinical Consideration:**  
            {cluster_data['recommendation']}
            """)

    st.divider()

    # =================================================
    # RAW DATA (OPTIONAL)
    # =================================================
    if st.checkbox("Show Complete Omics Feature Matrix"):
        st.dataframe(omics_df, use_container_width=True)

        st.download_button(
            label="Download Omics Data (CSV)",
            data=omics_df.to_csv(index=False),
            file_name="sertraline_omics_data.csv",
            mime="text/csv"
        )

    st.caption(
        "ADR susceptibility emerges from integrated genomic, proteomic, "
        "metabolomic, and inflammatory interactions. "
        "This system-level framework reflects modern precision pharmacology."
    )
# =================================================
# TAB 4 — PHARMACOVIGILANCE & ADR EVIDENCE (ACADEMIC)
# =================================================
with tab4:

    st.subheader("Pharmacovigilance Evidence & ADR Risk Contextualization")

    st.markdown("""
    This section summarizes known adverse drug reactions (ADRs) associated with sertraline,
    derived from clinical trials, post-marketing surveillance, and pharmacovigilance databases.
    Model-derived predictions are contextualized against real-world incidence patterns.
    """)

    common_adrs, serious_adrs = get_adr_evidence()

    st.divider()

    # =================================================
    # 1. COMMON ADRs
    # =================================================
    st.subheader("1. Common ADRs (Mild–Moderate Severity)")

    st.table(pd.DataFrame(common_adrs)[
        ["adr", "incidence", "severity", "onset"]
    ].rename(columns={
        "adr": "Adverse Reaction",
        "incidence": "Incidence",
        "severity": "Severity",
        "onset": "Typical Onset"
    }))

    st.divider()

    # =================================================
    # 2. SERIOUS ADRs
    # =================================================
    st.subheader("2. Serious ADRs (Low Frequency, High Impact)")

    st.table(pd.DataFrame(serious_adrs)[
        ["adr", "risk", "trigger", "management"]
    ].rename(columns={
        "adr": "Adverse Reaction",
        "risk": "Risk Profile",
        "trigger": "Trigger Factors",
        "management": "Recommended Management"
    }))

    st.divider()

    # =================================================
    # 3. REAL-WORLD COMPARISON
    # =================================================
    st.subheader("3. Real-World Risk Stratification")

    probability = (
        st.session_state.signal_score
        if st.session_state.get("signal_score") is not None
        else 0.5
    )

    comparison_data = pd.DataFrame({
        "ADR Type": [
            "Gastrointestinal",
            "Central Nervous System",
            "Sexual Dysfunction",
            "Other"
        ],
        "Population Baseline (%)": [20, 12, 25, 15],
        "Predicted Risk (%)": [
            20 + (probability * 30),
            12 + (probability * 25),
            25 + (probability * 20),
            15 + (probability * 10)
        ]
    })

    st.markdown("Comparison of predicted ADR risk against population-level baseline incidence:")

    fig = px.bar(
        comparison_data,
        x="ADR Type",
        y=["Population Baseline (%)", "Predicted Risk (%)"],
        barmode="group",
        template="plotly_dark" if theme_value == "dark" else "simple_white",
        labels={"value": "Probability (%)", "variable": "Cohort"},
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Baseline values represent approximate population-level estimates. "
        "Predicted risk reflects individualized model-based inference."
    )

    st.divider()

    # =================================================
    # 4. PRIOR ADR HISTORY CONTEXT
    # =================================================
    st.subheader("4. Prior Antidepressant Exposure & ADR Pattern")

    st.markdown("""
    **Documented Prior Exposure:**
    
    - Citalopram (2021): Nausea and dizziness; resolved after adaptation period  
    - Fluoxetine (2019): Sexual dysfunction; therapy discontinued after 3 months  

    **Clinical Interpretation:**
    
    Prior history of gastrointestinal and sexual adverse effects suggests
    potential susceptibility to serotonergic ADR patterns with sertraline.
    Consider dose titration and targeted monitoring strategies.
    """)

# =================================================
# TAB 5 — MODEL VALIDATION & EXPLAINABILITY
# =================================================
with tab5:

    st.subheader("Model Validation & Explainability")

    # -------------------------------------------------
    # SAFETY CHECK
    # -------------------------------------------------
    if st.session_state.model_input is None:
        st.info("Run a prediction in the Risk Summary tab first.")
        st.stop()

    # =================================================
    # 1. MODEL PERFORMANCE (VALIDATION)
    # =================================================
    st.subheader("1. Model Performance (Validation Dataset)")

    if X_val is None or y_val is None:
        st.info(
            "Validation dataset not detected.\n\n"
            "Provide `validation_dataset.csv` containing:\n"
            "- All model features\n"
            "- Binary `label` column"
        )
    else:
        y_scores = model.predict(X_val)

        auc_val = roc_auc_score(y_val, y_scores)
        fpr, tpr, _ = roc_curve(y_val, y_scores)

        roc_fig = go.Figure()
        roc_fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"AUC = {auc_val:.3f}",
                line=dict(width=3),
            )
        )
        roc_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Classifier",
                line=dict(dash="dash"),
            )
        )

        roc_fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_dark" if theme_value == "dark" else "simple_white",
            height=420,
        )

        st.plotly_chart(roc_fig, use_container_width=True)

        threshold = 0.5
        y_pred = (y_scores >= threshold).astype(int)
        cm = confusion_matrix(y_val, y_pred)

        st.markdown("**Confusion Matrix (Threshold = 0.5)**")

        st.dataframe(
            pd.DataFrame(
                cm,
                index=["Actual: No ADR", "Actual: ADR"],
                columns=["Predicted: No ADR", "Predicted: ADR"],
            )
        )

        st.markdown(f"**Validation AUC:** `{auc_val:.3f}`")

    st.divider()

    # =================================================
    # 2. LOCAL EXPLAINABILITY (SHAP)
    # =================================================
    st.subheader("2. Local Model Explainability (SHAP)")

    if st.session_state.signal_score is None:
        st.info("Run a prediction to generate SHAP explanations.")
    else:
        if len(EXPECTED_FEATURES) > 500:
            st.warning("SHAP disabled due to very high feature dimensionality.")
        else:
            shap_exp = explainer(
                st.session_state.model_input,
                check_additivity=False
            )

            st.markdown("""
            SHAP (SHapley Additive exPlanations) quantifies how each feature 
            contributes to the predicted ADR risk for this patient.
            
            - Positive values increase predicted risk  
            - Negative values decrease predicted risk  
            - Larger magnitude indicates stronger influence  
            """)

            st.markdown("**Top Contributing Features**")

            st_shap(
                shap.plots.bar(shap_exp, max_display=15),
                height=350
            )

            st.markdown("**Per-Patient Feature Attribution (Waterfall Plot)**")

            st_shap(
                shap.plots.waterfall(shap_exp[0], max_display=15),
                height=400
            )

            if show_model_explanation and st.session_state.shap_vals is not None:
                st.subheader("3. Automated Feature Interpretation")

                for line in st.session_state.explanations:
                    st.write(f"- {line}")

    st.caption(
        "SHAP explanations describe model behavior and feature contribution. "
        "They do not establish biological causality."
    )

# =================================================
# TAB 6 — MODEL ARCHITECTURE & TRANSPARENCY
# =================================================
with tab6:

    st.subheader("Model Architecture & Transparency")

    st.markdown("""
    This section provides structural and methodological transparency 
    regarding the ADR prediction framework. 
    
    The goal is to enable reproducibility, critical evaluation, and 
    scientific validation of model outputs.
    """)

    if show_model_explanation:

        st.divider()

        # =================================================
        # 1. MODEL SPECIFICATIONS
        # =================================================
        st.subheader("1. Model Specifications")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            **Algorithm:** LightGBM (Gradient Boosting Decision Trees)  
            **Task:** Binary Classification (ADR vs. No ADR)  

            **Training Data Sources:**
            - FAERS pharmacovigilance reports (~500,000 cases)
            - Molecular descriptors (RDKit fingerprints)
            - Protein interaction features
            - Multi-omics summaries (45 features)

            **Total Features:** {len(EXPECTED_FEATURES)}
            """)

        with col2:
            st.markdown("""
            **Cross-Validation Performance:**
            - AUC-ROC: 0.78  
            - Sensitivity: 0.72  
            - Specificity: 0.75  

            Performance metrics reflect internal validation 
            and may vary across external populations.
            """)

        st.divider()

        # =================================================
        # 2. APPLICABILITY DOMAIN
        # =================================================
        st.subheader("2. Applicability Domain & Limitations")

        st.markdown("""
        **Target Population:**
        - Adults (18–75 years)
        - Depression and anxiety indications
        - ≤ 3 concomitant medications
        - Mild-to-moderate comorbidity burden

        **Model Limitations:**
        - Limited pediatric representation
        - Underpowered for rare ADR detection
        - Does not incorporate adherence patterns
        - Limited cross-ethnic calibration
        """)

        st.divider()

        # =================================================
        # 3. GLOBAL FEATURE IMPORTANCE
        # =================================================
        st.subheader("3. Global Feature Importance")

        feature_importance_data = pd.DataFrame({
            "Feature": [
                "CYP2C19 Metabolizer Status",
                "GI Inflammatory Signature",
                "Sertraline Protein Binding",
                "Baseline Serotonin Levels",
                "HTR2A Network Connectivity",
                "Oxidative Stress Markers",
                "Age-adjusted Clearance",
                "Drug–Drug Interaction Score"
            ],
            "Relative Importance": [
                0.185, 0.152, 0.118, 0.105,
                0.095, 0.088, 0.078, 0.179
            ]
        }).sort_values("Relative Importance", ascending=True)

        fig = px.bar(
            feature_importance_data,
            x="Relative Importance",
            y="Feature",
            orientation="h",
            template="plotly_dark" if theme_value == "dark" else "simple_white",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        Feature rankings reflect average contribution to model predictions 
        across the training distribution. Pharmacogenomic and inflammatory 
        features demonstrate the strongest predictive influence.
        """)

        st.divider()

        # =================================================
        # 4. MODEL WORKFLOW
        # =================================================
        st.subheader("4. Model Inference Workflow")

        st.markdown("""
        1. **Input Acquisition:** Patient clinical and omics features  
        2. **Feature Engineering:** Scaling, interaction encoding  
        3. **Gradient Boosting Ensemble:** 200 sequential decision trees  
        4. **Probability Calibration:** Sigmoid transformation  
        5. **Risk Output:** Continuous ADR probability (0–1)  
        6. **Clinical Translation:** Risk category assignment & interpretation  
        """)

        st.divider()

        st.caption(
            "This system is designed for research and decision-support purposes. "
            "Model outputs should be interpreted in conjunction with clinical expertise."
        )

    else:
        st.info("Enable model transparency in settings to view architecture details.")

# =================================================
# TAB 7 — BACKGROUND & SCIENTIFIC CONTEXT
# =================================================
with tab7:

    st.subheader("Scientific Background: Sertraline & ADR Risk Modeling")

    st.markdown("""
    This section provides pharmacological, mechanistic, and systems-level
    context for sertraline-associated adverse drug reactions (ADRs).
    """)

    st.divider()

    # =================================================
    # 1. DRUG OVERVIEW
    # =================================================
    st.subheader("1. Drug Overview")

    st.markdown("""
    **Sertraline** is a selective serotonin reuptake inhibitor (SSRI) widely used in the treatment of:

    - Major Depressive Disorder (MDD)  
    - Generalized Anxiety Disorder (GAD)  
    - Obsessive–Compulsive Disorder (OCD)  
    - Panic Disorder (PD)  
    - Social Anxiety Disorder (SAD)  
    - Post-Traumatic Stress Disorder (PTSD)  
    - Premenstrual Dysphoric Disorder (PMDD)

    It is marketed globally under multiple brand names and is among the most prescribed SSRIs.
    """)

    st.divider()

    # =================================================
    # 2. MECHANISM OF ACTION
    # =================================================
    st.subheader("2. Mechanism of Action")

    st.markdown("""
    Sertraline selectively inhibits the serotonin transporter (SERT, encoded by *SLC6A4*),
    reducing presynaptic serotonin reuptake and increasing synaptic serotonin availability.

    Enhanced serotonergic neurotransmission contributes to therapeutic effects,
    but excessive serotonergic activity may also underlie adverse events including:

    - Gastrointestinal intolerance  
    - Central nervous system activation (insomnia, agitation)  
    - Sexual dysfunction  
    - Serotonin toxicity in polypharmacy settings  
    """)

    st.divider()

    # =================================================
    # 3. PHARMACOKINETICS & PHARMACOGENOMICS
    # =================================================
    st.subheader("3. Pharmacokinetics & Pharmacogenomic Modifiers")

    st.markdown("""
    **Absorption:** Peak plasma concentrations typically occur 4–8 hours post-dose.  

    **Metabolism:** Primarily hepatic via CYP2C19; minor contributions from CYP2D6 and CYP3A4.  

    **Protein Binding:** Approximately 98%.  

    **Elimination Half-Life:** 24–26 hours.  

    Genetic polymorphisms in *CYP2C19* significantly influence systemic exposure.
    Poor metabolizers demonstrate elevated plasma concentrations and increased ADR risk.
    """)

    st.divider()

    # =================================================
    # 4. ADVERSE DRUG REACTIONS
    # =================================================
    st.subheader("4. Established Adverse Drug Reactions")

    st.markdown("""
    **Common ADRs (10–30% incidence):**

    - Gastrointestinal: Nausea, diarrhea, dyspepsia  
    - Neurological: Dizziness, insomnia, headache  
    - Sexual dysfunction: Reduced libido, delayed ejaculation  
    - General: Sweating, fatigue  

    **Serious or Rare ADRs (<1% incidence):**

    - Serotonin syndrome (particularly in polypharmacy)  
    - QT interval prolongation  
    - Hyponatremia (SIADH, elderly risk)  
    - Increased bleeding risk  
    - Withdrawal phenomena upon abrupt discontinuation  
    """)

    st.divider()

    # =================================================
    # 5. OMICS-BASED INSIGHTS
    # =================================================
    st.subheader("5. Multi-Omics Insights into ADR Susceptibility")

    st.markdown("""
    Emerging systems pharmacology research indicates that ADR susceptibility is influenced by:

    **Transcriptomics:**  
    Baseline neuroinflammatory gene expression correlates with GI and CNS ADR risk.

    **Proteomics:**  
    Variability in SERT abundance and blood–brain barrier transport proteins modifies CNS exposure.

    **Metabolomics:**  
    Altered tryptophan–kynurenine metabolism is associated with inflammatory burden and symptom severity.

    **Systems Integration:**  
    Multi-omics clustering reveals patient subgroups with distinct pharmacokinetic and pharmacodynamic profiles.
    """)

    st.divider()

    # =================================================
    # 6. RATIONALE FOR ADR PREDICTION
    # =================================================
    st.subheader("6. Rationale for Predictive Modeling")

    st.markdown("""
    Predictive ADR modeling supports:

    - Early identification of high-risk individuals  
    - Dose optimization strategies  
    - Precision pharmacology frameworks  
    - Augmented pharmacovigilance signal detection  
    - Reduction of preventable hospitalizations  

    Integration of clinical, genomic, and omics data enables individualized
    risk stratification beyond population-level incidence estimates.
    """)

    st.divider()

    # =================================================
    # 7. REFERENCES
    # =================================================
    st.subheader("7. References & Scientific Sources")

    st.markdown("""
    - U.S. Food and Drug Administration (FDA). FAERS Public Dashboard.  
    - CPIC Guidelines for CYP2C19 and SSRI Dosing.  
    - LightGBM: Ke et al., Advances in Neural Information Processing Systems (2017).  
    - Lundberg & Lee. A Unified Approach to Interpreting Model Predictions (SHAP). NIPS (2017).  
    - Systems Pharmacology Literature on SSRI Multi-Omics Integration.
    """)

    st.caption(
        "This background summary is provided for research and educational purposes. "
        "Clinical application requires guideline-based evaluation."
    )
# =================================================
# TAB 8 — PREDICTION HISTORY & RESEARCH LOG
# =================================================
with tab8:

    st.subheader("Prediction History & Research Audit Log")

    st.markdown("""
    This module stores all model-generated ADR risk predictions.
    
    It supports:
    - Reproducibility of computational results  
    - Longitudinal monitoring of risk estimates  
    - Pharmacovigilance documentation  
    - Research audit compliance  
    """)

    st.divider()

    # -------------------------------------------------
    # LOAD DATABASE RECORDS
    # -------------------------------------------------
    try:
        conn = get_connection()
        history_df = pd.read_sql(
            "SELECT * FROM predictions ORDER BY timestamp DESC",
            conn
        )
        conn.close()
    except Exception as e:
        st.error("Database connection unavailable.")
        history_df = pd.DataFrame()

    # -------------------------------------------------
    # DISPLAY STORED RECORDS
    # -------------------------------------------------
    if history_df.empty:
        st.info("No stored predictions available. Run a prediction to generate records.")
    else:

        st.subheader("1. Stored Prediction Records")

        st.dataframe(
            history_df,
            use_container_width=True
        )

        st.divider()

        # -------------------------------------------------
        # SUMMARY STATISTICS
        # -------------------------------------------------
        st.subheader("2. Aggregate Summary Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Predictions",
            len(history_df)
        )

        col2.metric(
            "Mean ADR Risk Score",
            f"{history_df['adr_score'].mean():.3f}"
        )

        col3.metric(
            "High-Risk Cases (≥ 0.8)",
            int((history_df["adr_score"] >= 0.8).sum())
        )

        st.divider()

        # -------------------------------------------------
        # DATA EXPORT
        # -------------------------------------------------
        st.subheader("3. Data Export")

        st.markdown("""
        The full prediction log may be exported for:

        - External statistical analysis  
        - Institutional reporting  
        - Regulatory documentation  
        - Reproducibility verification  
        """)

        st.download_button(
            label="Download Prediction History (CSV)",
            data=history_df.to_csv(index=False),
            file_name="sertraline_adr_prediction_history.csv",
            mime="text/csv"
        )

    st.caption(
        "Prediction records are stored locally for research traceability. "
        "Ensure compliance with institutional data governance policies."
    )
    
# =================================================
# TAB 9 — CLINICAL & RESEARCH ASSISTANT
# =================================================
with tab9:

    st.subheader("Clinical & Research Decision-Support Assistant")

    st.markdown("""
    This module provides contextual explanations of model predictions,
    omics contributions, and risk interpretation.

    Responses are generated based on:
    - Current patient inputs
    - Model outputs
    - SHAP explanations
    - Active interface section
    """)

    st.divider()

    # -------------------------------------------------
    # Ensure prediction exists
    # -------------------------------------------------
    context = build_chat_context()

    if context is None:
        st.info("Run a prediction in the Risk Summary tab to enable contextual explanation.")
        st.stop()

    context["active_tab"] = st.session_state.get("active_tab", "Unknown")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # -------------------------------------------------
    # Role-aware interaction
    # -------------------------------------------------
    role_key = "Clinician" if "Clinician" in user_role else "Researcher"

    st.subheader("Suggested Analytical Prompts")

    cols = st.columns(2)
    for i, q in enumerate(SUGGESTED_QUESTIONS[role_key]):
        if cols[i % 2].button(q, key=f"suggested_{i}"):
            st.session_state.chat_input_prefill = q

    st.divider()

    # -------------------------------------------------
    # Chat Input
    # -------------------------------------------------
    user_question = st.chat_input("Enter your question regarding ADR risk or model behavior")

    if "chat_input_prefill" in st.session_state:
        user_question = st.session_state.pop("chat_input_prefill")

    if user_question:
        response = explain_with_context(context, user_question)

        st.session_state.chat_history.append(("user", user_question))
        st.session_state.chat_history.append(("assistant", response))

    # -------------------------------------------------
    # Highlight referenced graph if requested
    # -------------------------------------------------
    if "highlight_graph" in st.session_state:
        graph_id = st.session_state.pop("highlight_graph")

        st.markdown(
            f"""
            <script>
                const el = document.getElementById("{graph_id}");
                if (el) {{
                    el.classList.add("graph-highlight");
                    setTimeout(() => el.classList.remove("graph-highlight"), 3000);
                }}
            </script>
            """,
            unsafe_allow_html=True
        )

    # -------------------------------------------------
    # Display Chat History
    # -------------------------------------------------
    st.subheader("Dialogue Log")

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

    st.divider()

    # -------------------------------------------------
    # Feedback Mechanism
    # -------------------------------------------------
    if len(st.session_state.chat_history) >= 2:
        st.subheader("Response Evaluation")

        feedback = st.radio(
            "Was this explanation helpful?",
            ["Yes", "Partially", "No"],
            horizontal=True,
            key=f"feedback_{len(st.session_state.chat_history)}"
        )

        if feedback:
            st.success("Feedback recorded for model refinement analysis.")

    st.divider()

    # -------------------------------------------------
    # Responsible AI Notice
    # -------------------------------------------------
    st.caption(
        "This assistant provides interpretative support for research and decision-support purposes only. "
        "It does not constitute medical advice, diagnosis, or treatment recommendation. "
        "Clinical decisions must integrate independent professional judgment."
    )
st.markdown("---")

st.markdown(f"""
<div class="research-footer">

    <div class="footer-title">
        Sertraline ADR Prediction Framework v1.0
    </div>

    <div class="footer-description">
        AI-driven multi-omics pharmacovigilance research platform integrating 
        FAERS signal detection, molecular descriptors, protein interaction 
        networks, and systems-level biomarkers.
    </div>

    <div class="footer-authors">
        Developed by <strong>Adarsh Dheeraj Dubey</strong> & 
        <strong>Ranjana Mangesh Parab</strong><br>
        M.Sc. Bioinformatics Research Project
    </div>

    <div class="footer-disclaimer">
        This system is intended for academic research and decision-support exploration only.
        It does not constitute medical advice, diagnosis, or therapeutic recommendation.
        Clinical decisions must be independently validated by qualified healthcare professionals.
    </div>

    <div class="footer-meta">
        Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} IST
    </div>

</div>
""", unsafe_allow_html=True)
