import pandas as pd
import numpy as np

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
RAW_DATA_PATH = "data/sertraline_adr_faers_rdkit_omics.csv"
OUT_PATH = "data/validation_dataset.csv"

# -------------------------------------------------
# LOAD RAW DATA
# -------------------------------------------------
df = pd.read_csv(RAW_DATA_PATH)

print("📦 Loaded data shape:", df.shape)

# -------------------------------------------------
# TARGET (same as training)
# -------------------------------------------------
y = df["label"].astype(int)

# -------------------------------------------------
# BASE FEATURES (NUMERIC ONLY)
# -------------------------------------------------
X = df.select_dtypes(include=[np.number]).drop(columns=["label"])

# -------------------------------------------------
# REMOVE LEAKY FEATURES (CRITICAL)
# -------------------------------------------------
LEAKY_FEATURES = [
    "PRR", "ROR", "logPRR", "logROR", "a_count", "total_count"
]

X = X.drop(
    columns=[c for c in LEAKY_FEATURES if c in X.columns],
    errors="ignore"
)

print("🧹 Removed leaky features")

# -------------------------------------------------
# LEVEL-2 OMICS (BIOLOGY-GUIDED)
# -------------------------------------------------
X["neuroinflammation_score"] = (
    X["MolLogP"] * 0.3 + X["HeavyAtomMolWt"] * 0.001
)

X["oxidative_stress_score"] = (
    X["NumValenceElectrons"] * 0.01
)

X["bbb_integrity_score"] = 1 / (1 + X["MolLogP"])

X["cytokine_activation_score"] = (
    X["TPSA"] * 0.02
)

# -------------------------------------------------
# PHARMACOGENOMICS (POPULATION-LEVEL PROXIES)
# -------------------------------------------------
X["cyp2c19_activity_score"] = (
    1 / (1 + X["MolLogP"])
)

X["cyp2d6_activity_score"] = (
    X["NumHDonors"] + 1
)

X["sert_expression_score"] = (
    X["TPSA"] * 0.01
)

# -------------------------------------------------
# INTERACTION FEATURES
# -------------------------------------------------
if "dose_mg" not in X.columns:
    X["dose_mg"] = np.random.choice([25, 50, 100, 150], len(X))

if "polypharmacy_flag" not in X.columns:
    X["polypharmacy_flag"] = np.random.binomial(1, 0.3, len(X))

X["metabolic_overload_score"] = (
    np.log1p(X["dose_mg"])
    * (1 / (X["cyp2c19_activity_score"] + 0.1))
    * (1 + X["polypharmacy_flag"])
)

X["cns_adr_risk_score"] = (
    X["neuroinflammation_score"]
    * (1 / (X["bbb_integrity_score"] + 0.01))
    * X["MolLogP"]
)

X["gi_immune_adr_score"] = (
    X["cytokine_activation_score"]
    * X["oxidative_stress_score"]
)

# -------------------------------------------------
# FINAL VALIDATION DATASET
# -------------------------------------------------
val_df = X.copy()
val_df["label"] = y.values

val_df.to_csv(OUT_PATH, index=False)

print("✅ Validation dataset saved:", OUT_PATH)
print("📊 Final shape:", val_df.shape)
print("📊 Positive class ratio:", y.mean())
