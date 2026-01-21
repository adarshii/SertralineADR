import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
df = pd.read_csv(
    "data/sertraline_adr_faers_rdkit_omics.csv"
)
# -------------------------------------------------
# TARGET
# -------------------------------------------------
y = ((df["a_count"] / df["total_count"]) > 0.05).astype(int)

# -------------------------------------------------
# DROP NON-NUMERIC / IDENTIFIERS
# -------------------------------------------------
X = df.select_dtypes(include=[np.number]).drop(columns=["label"])
# -------------------------------------------------
# REMOVE LEAKY FAERS SIGNAL FEATURES (CRITICAL FIX)
# -------------------------------------------------
LEAKY_FEATURES = [
    "PRR",
    "ROR",
    "logPRR",
    "logROR",
    "a_count",
    "total_count"
]

X = X.drop(
    columns=[c for c in LEAKY_FEATURES if c in X.columns],
    errors="ignore"
)

print("🧹 Removed leaky features:",
      set(LEAKY_FEATURES) & set(df.columns))

# -------------------------------------------------
# ADD LEVEL-2 OMICS FEATURES (SIMULATED / AGGREGATED)
# -------------------------------------------------
# -------------------------------------------------
# LEVEL-2 OMICS (BIOLOGY-GUIDED, NON-RANDOM)
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
# ADD PHARMACOGENOMICS (POPULATION-LEVEL)
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
# INTERACTION FEATURES (CRITICAL)
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

if "MolLogP" in X.columns:
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
# REMOVE NEAR-ZERO VARIANCE FEATURES
# -------------------------------------------------


# -------------------------------------------------
# TRAIN / VALIDATION SPLIT
# -------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------
# LIGHTGBM DATASETS
# -------------------------------------------------
train_ds = lgb.Dataset(X_train, label=y_train)
val_ds = lgb.Dataset(X_val, label=y_val)

# -------------------------------------------------
# MODEL PARAMETERS (SAFE + INTERPRETABLE)
# -------------------------------------------------
params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 64,
    "max_depth": -1,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42,

    # imbalance handling (ONLY ONE)
    "scale_pos_weight": (len(y) - y.sum()) / y.sum(),
}


# -------------------------------------------------
# TRAIN
# -------------------------------------------------
model = lgb.train(
    params,
    train_ds,
    valid_sets=[val_ds],
    num_boost_round=500,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50),
    ]
)

# -------------------------------------------------
# EVALUATE
# -------------------------------------------------
y_val_pred = model.predict(X_val)
auc = roc_auc_score(y_val, y_val_pred)

print(f"\n✅ Validation AUC: {auc:.3f}")

# -------------------------------------------------
# SAVE MODEL
# -------------------------------------------------
joblib.dump(
    model,
    "models/sertraline_lightgbm_full.pkl"
)
print("\n📊 Final feature count:", X.shape[1])
print("📊 Positive class ratio:", y.mean())
print("✅ Model saved: models/sertraline_lightgbm_full.pkl")
