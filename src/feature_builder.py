# ================================================
# feature_builder.py
# ================================================
# Builds a SINGLE inference-ready feature vector
# EXACTLY aligned with the trained LightGBM model
# ================================================

import pandas as pd
import numpy as np

# ------------------------------------------------
# Build feature vector
# ------------------------------------------------
def build_feature_vector(
    expected_features,
    age,
    sex,
    dose,
    polypharmacy,
    liver_disease,
    neuroinflammation,
    oxidative_stress,
    bbb_integrity,
    cytokine_activity,
    cyp2c19_status,
    cyp2d6_status,
    sert_expression,
):
    """
    Returns a 1-row DataFrame with EXACTLY the same
    feature names and order as the trained model.
    """

    # --------------------------------------------
    # 1. Initialize empty feature frame
    # --------------------------------------------
    X = pd.DataFrame(
        0.0,
        index=[0],
        columns=expected_features
    )

    # --------------------------------------------
    # 2. CLINICAL BASE FEATURES (if present)
    # --------------------------------------------
    if "age" in X.columns:
        X["age"] = age

    if "dose_mg" in X.columns:
        X["dose_mg"] = dose

    if "female" in X.columns:
        X["female"] = 1 if sex == "Female" else 0

    if "polypharmacy_flag" in X.columns:
        X["polypharmacy_flag"] = int(polypharmacy)

    if "liver_disease_flag" in X.columns:
        X["liver_disease_flag"] = int(liver_disease)

    # --------------------------------------------
    # 3. LEVEL-2 OMICS (MATCH TRAINING FORMULAS)
    # --------------------------------------------
    # These overwrite defaults ONLY if columns exist

    if "neuroinflammation_score" in X.columns:
        X["neuroinflammation_score"] = neuroinflammation

    if "oxidative_stress_score" in X.columns:
        X["oxidative_stress_score"] = oxidative_stress

    if "bbb_integrity_score" in X.columns:
        X["bbb_integrity_score"] = bbb_integrity

    if "cytokine_activation_score" in X.columns:
        X["cytokine_activation_score"] = cytokine_activity

    # --------------------------------------------
    # 4. PHARMACOGENOMICS ENCODING (DETERMINISTIC)
    # --------------------------------------------
    cyp_map = {
        "Poor": 0.25,
        "Intermediate": 0.5,
        "Normal": 1.0,
        "Ultra-rapid": 1.5,
    }

    sert_map = {
        "Low": 0.5,
        "Normal": 1.0,
        "High": 2.0,
    }

    if "cyp2c19_activity_score" in X.columns:
        X["cyp2c19_activity_score"] = cyp_map[cyp2c19_status]

    if "cyp2d6_activity_score" in X.columns:
        X["cyp2d6_activity_score"] = cyp_map[cyp2d6_status]

    if "sert_expression_score" in X.columns:
        X["sert_expression_score"] = sert_map[sert_expression]

    # --------------------------------------------
    # 5. INTERACTION FEATURES (MATCH TRAINING)
    # --------------------------------------------

    # Metabolic overload
    if all(
        c in X.columns
        for c in [
            "dose_mg",
            "cyp2c19_activity_score",
            "polypharmacy_flag",
        ]
    ):
        X["metabolic_overload_score"] = (
            np.log1p(X["dose_mg"])
            * (1 / (X["cyp2c19_activity_score"] + 0.1))
            * (1 + X["polypharmacy_flag"])
        )

    # CNS ADR risk
    if all(
        c in X.columns
        for c in [
            "neuroinflammation_score",
            "bbb_integrity_score",
            "MolLogP",
        ]
    ):
        X["cns_adr_risk_score"] = (
            X["neuroinflammation_score"]
            * (1 / (X["bbb_integrity_score"] + 0.01))
            * X["MolLogP"]
        )

    # GI / immune ADR risk
    if all(
        c in X.columns
        for c in [
            "cytokine_activation_score",
            "oxidative_stress_score",
        ]
    ):
        X["gi_immune_adr_score"] = (
            X["cytokine_activation_score"]
            * X["oxidative_stress_score"]
        )
    # --- HARD-CODE SERTRALINE RDKit DESCRIPTORS ---
    SERTRALINE_CONSTANTS = {
        "MolLogP": 5.1,
        "MolWt": 306.23,
        "TPSA": 12.5,
        "NumHDonors": 1,
        "NumHAcceptors": 1,
        "NumRotatableBonds": 4,
        }

    for k, v in SERTRALINE_CONSTANTS.items():
        if k in X.columns:
            X[k] = v


    # --------------------------------------------
    # 6. FINAL SAFETY CHECK
    # --------------------------------------------
    assert (
        X.shape[1] == len(expected_features)
    ), "Feature count mismatch with trained model"

    return X
