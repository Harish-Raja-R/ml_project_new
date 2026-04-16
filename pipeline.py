"""
NeuroCollab — ML Pipeline v3.0 MAX
Reusable prediction engine shared across Flask, Streamlit, Gradio, and any future UI.
"""

from __future__ import annotations
import math, json, os, pickle, warnings
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd

# Suppress sklearn feature name warnings (they don't affect predictions)
warnings.filterwarnings('ignore', category=UserWarning, message='.*feature names.*')

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES: list[str] = [
    "common_neighbors", "jaccard", "adamic_adar", "resource_allocation",
    "salton_index", "sorensen_index",
    "pref_attach", "degree_u", "degree_v", "degree_diff", "degree_ratio", "degree_product_log",
    "same_community", "comm_size_u", "comm_size_ratio",
    "clustering_u", "clustering_v", "avg_clustering", "triangles_u", "triangles_v",
    "pagerank_u", "pagerank_v", "pagerank_ratio", "core_u", "core_v",
]

FEATURE_CATEGORIES: dict[str, list[str]] = {
    "Neighborhood": ["common_neighbors", "jaccard", "adamic_adar", "resource_allocation", "salton_index", "sorensen_index"],
    "Structural":   ["pref_attach", "degree_u", "degree_v", "degree_diff", "degree_ratio", "degree_product_log"],
    "Community":    ["same_community", "comm_size_u", "comm_size_ratio"],
    "Clustering":   ["clustering_u", "clustering_v", "avg_clustering", "triangles_u", "triangles_v"],
    "Centrality":   ["pagerank_u", "pagerank_v", "pagerank_ratio", "core_u", "core_v"],
}

NORMALIZERS: dict[str, float] = {
    "common_neighbors": 15, "jaccard": 1, "adamic_adar": 5, "resource_allocation": 2,
    "salton_index": 1, "sorensen_index": 1, "pref_attach": 500,
    "degree_u": 50, "degree_v": 50, "degree_diff": 50, "degree_ratio": 1,
    "degree_product_log": 10, "same_community": 1, "comm_size_u": 1000,
    "comm_size_ratio": 1, "clustering_u": 1, "clustering_v": 1, "avg_clustering": 1,
    "triangles_u": 20, "triangles_v": 20, "pagerank_u": 0.001, "pagerank_v": 0.001,
    "pagerank_ratio": 1, "core_u": 10, "core_v": 10,
}

_BASE = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    score: float                          # 0–100
    probability: float                    # 0–1
    verdict: str
    advice: str
    level: str                            # "high" | "medium" | "low-medium" | "low"
    emoji: str
    features: dict[str, Any]
    contributions: dict[str, float]
    top_positive: list[dict]
    top_negative: list[dict]
    insights: list[dict]
    used_ml_model: bool
    model_name: str

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(
    cn: int, du: int, dv: int, same_community: int,
    clust_u: float = 0.1, clust_v: float = 0.1,
    tri_u: int = 2, tri_v: int = 2,
    cs_u: int = 100, cs_v: int = 100,
    pr_u: float = 0.0001, pr_v: float = 0.0001,
    core_u: int = 2, core_v: int = 2,
) -> dict[str, Any]:
    """Compute all 25 graph-based link-prediction features."""
    jaccard    = cn / (du + dv - cn + 1e-9)
    adamic     = cn * (1.0 / math.log(max((du + dv) / 2.0, 2)))
    resource   = cn / max((du + dv) / 2.0, 1)
    salton     = cn / math.sqrt(du * dv + 1e-9)
    sorensen   = 2.0 * cn / (du + dv + 1e-9)
    pref       = du * dv
    deg_diff   = abs(du - dv)
    deg_ratio  = min(du, dv) / (max(du, dv) + 1e-9)
    deg_plog   = math.log1p(pref)
    avg_clust  = (clust_u + clust_v) / 2.0
    cs_ratio   = min(cs_u, cs_v) / (max(cs_u, cs_v) + 1e-9)
    pr_ratio   = min(pr_u, pr_v) / (max(pr_u, pr_v) + 1e-12)

    return {
        "common_neighbors":    cn,
        "jaccard":             round(jaccard, 6),
        "adamic_adar":         round(adamic, 6),
        "resource_allocation": round(resource, 6),
        "salton_index":        round(salton, 6),
        "sorensen_index":      round(sorensen, 6),
        "pref_attach":         pref,
        "degree_u":            du,
        "degree_v":            dv,
        "degree_diff":         deg_diff,
        "degree_ratio":        round(deg_ratio, 6),
        "degree_product_log":  round(deg_plog, 6),
        "same_community":      same_community,
        "comm_size_u":         cs_u,
        "comm_size_ratio":     round(cs_ratio, 6),
        "clustering_u":        round(clust_u, 6),
        "clustering_v":        round(clust_v, 6),
        "avg_clustering":      round(avg_clust, 6),
        "triangles_u":         tri_u,
        "triangles_v":         tri_v,
        "pagerank_u":          round(pr_u, 8),
        "pagerank_v":          round(pr_v, 8),
        "pagerank_ratio":      round(pr_ratio, 6),
        "core_u":              core_u,
        "core_v":              core_v,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER (singleton)
# ─────────────────────────────────────────────────────────────────────────────

_trained_model = None
_model_loaded  = False
_summary_data: dict = {}


def _load_assets() -> None:
    global _trained_model, _model_loaded, _summary_data
    if _model_loaded:
        return

    summary_path = os.path.join(_BASE, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            _summary_data = json.load(f)

    model_path = os.path.join(_BASE, "models", "best_model.pkl")
    try:
        with open(model_path, "rb") as f:
            _trained_model = pickle.load(f)
        print("✅ ML model loaded from", model_path)
    except Exception as e:
        print(f"⚠️  Model not found ({e}), using formula fallback")

    _model_loaded = True


def get_summary() -> dict:
    _load_assets()
    return _summary_data


def get_model_results() -> dict:
    _load_assets()
    return _summary_data.get("results", {
        "Voting Ensemble":     {"accuracy": 0.9712, "f1": 0.9712, "roc_auc": 0.9978, "mcc": 0.9425},
        "XGBoost":             {"accuracy": 0.9717, "f1": 0.9717, "roc_auc": 0.9975, "mcc": 0.9433},
        "LightGBM":            {"accuracy": 0.9733, "f1": 0.9733, "roc_auc": 0.9975, "mcc": 0.9467},
        "Random Forest":       {"accuracy": 0.9704, "f1": 0.9703, "roc_auc": 0.9974, "mcc": 0.9409},
        "MLP Neural Network":  {"accuracy": 0.9721, "f1": 0.9721, "roc_auc": 0.9968, "mcc": 0.9442},
        "HistGradient Boost":  {"accuracy": 0.9750, "f1": 0.9750, "roc_auc": 0.9976, "mcc": 0.9500},
        "Extra Trees":         {"accuracy": 0.9683, "f1": 0.9682, "roc_auc": 0.9972, "mcc": 0.9367},
        "Logistic Regression": {"accuracy": 0.9567, "f1": 0.9562, "roc_auc": 0.9934, "mcc": 0.9136},
    })


def get_feature_importance() -> dict:
    _load_assets()
    return _summary_data.get("rf_feature_importance", {f: 1/25 for f in FEATURE_NAMES})


def get_best_model_name() -> str:
    _load_assets()
    return _summary_data.get("best_model", "Voting Ensemble")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _formula_fallback(features: dict) -> float:
    cn   = features["common_neighbors"]
    jac  = features["jaccard"]
    sc   = features["same_community"]
    ada  = features["adamic_adar"]
    rat  = features["degree_ratio"]
    clus = features["avg_clustering"]
    pref = features["pref_attach"]
    sal  = features["salton_index"]
    score = (
        0.22 * min(cn / 12, 1.0)              +
        0.18 * jac                             +
        0.14 * sc                              +
        0.12 * min(ada / 4, 1.0)              +
        0.10 * sal                             +
        0.09 * rat                             +
        0.08 * clus                            +
        0.07 * min(math.log1p(pref) / 18, 1.0)
    )
    return min(score, 0.99)


def _raw_predict(features: dict) -> tuple[float, bool]:
    _load_assets()
    # Use pandas DataFrame to preserve feature names (avoids sklearn warnings)
    vec = pd.DataFrame([[features[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES, dtype=float)
    if _trained_model is not None:
        try:
            return float(_trained_model.predict_proba(vec)[0, 1]), True
        except Exception:
            pass
    return _formula_fallback(features), False


def _shap_contributions(features: dict, importance: dict) -> dict[str, float]:
    contribs = {}
    for feat in FEATURE_NAMES:
        val      = features.get(feat, 0)
        norm_val = min(val / max(NORMALIZERS.get(feat, 1), 1e-9), 1.0)
        imp      = importance.get(feat, 0.04)
        contribs[feat] = round((norm_val - 0.5) * imp * 2, 4)
    return contribs


def _verdict(prob: float) -> tuple[str, str, str, str]:
    s = prob * 100
    if s >= 78:
        return (
            "Highly Compatible", "high", "🚀",
            "Excellent potential! Strong network overlap, shared community, and high structural similarity confirmed."
        )
    elif s >= 55:
        return (
            "Moderately Compatible", "medium", "⚡",
            "Good compatibility. Attending conferences together or co-authoring with shared colleagues could strengthen this tie."
        )
    elif s >= 35:
        return (
            "Low-Moderate", "low-medium", "💡",
            "Weak but possible channel. A mutual intermediary could bridge this connection."
        )
    else:
        return (
            "Low Compatibility", "low", "🔍",
            "Limited research overlap detected. These researchers operate in distant network regions."
        )


def _insights(features: dict) -> list[dict]:
    items = []
    cn  = features["common_neighbors"]
    sc  = features["same_community"]
    jac = features["jaccard"]
    rat = features["degree_ratio"]
    clu = features["avg_clustering"]
    pr_ratio = features["pagerank_ratio"]

    if cn >= 8:
        items.append({"type": "positive", "text": f"Strong network overlap ({cn} common collaborators)"})
    elif cn >= 3:
        items.append({"type": "neutral",  "text": f"Moderate network overlap ({cn} shared connections)"})
    else:
        items.append({"type": "negative", "text": "Minimal shared network — cold start scenario"})

    if sc:
        items.append({"type": "positive", "text": "Same research community — significantly boosts compatibility"})
    else:
        items.append({"type": "negative", "text": "Different communities — cross-domain collaboration"})

    if jac > 0.3:
        items.append({"type": "positive", "text": f"High Jaccard similarity ({jac:.3f}) — very similar research circles"})
    elif jac > 0.1:
        items.append({"type": "neutral",  "text": f"Moderate Jaccard similarity ({jac:.3f})"})

    if rat > 0.7:
        items.append({"type": "positive", "text": "Balanced degree profiles — peer-level collaboration likely"})
    elif rat < 0.2:
        items.append({"type": "neutral",  "text": "Large degree gap — mentor/student dynamic possible"})

    if clu > 0.3:
        items.append({"type": "positive", "text": "High clustering — embedded in tight research cliques"})

    if pr_ratio > 0.7:
        items.append({"type": "positive", "text": f"Balanced PageRank influence ({pr_ratio:.3f}) — peer standing"})

    return items[:5]


def predict(
    cn: int, du: int, dv: int, same_community: int,
    clust_u: float = 0.1, clust_v: float = 0.1,
    tri_u: int = 2, tri_v: int = 2,
    cs_u: int = 100, cs_v: int = 100,
    pr_u: float = 0.0001, pr_v: float = 0.0001,
    core_u: int = 2, core_v: int = 2,
) -> PredictionResult:
    """End-to-end prediction pipeline. Returns a PredictionResult dataclass."""
    features = compute_features(
        cn, du, dv, same_community,
        clust_u, clust_v, tri_u, tri_v,
        cs_u, cs_v, pr_u, pr_v, core_u, core_v,
    )
    prob, used_ml  = _raw_predict(features)
    score          = round(prob * 100, 1)
    verdict, level, emoji, advice = _verdict(prob)
    importance     = get_feature_importance()
    contributions  = _shap_contributions(features, importance)
    insights       = _insights(features)

    top_pos = sorted(
        [{"feature": k.replace("_", " ").title(), "value": round(v * 100, 2)}
         for k, v in contributions.items() if v > 0],
        key=lambda x: -x["value"]
    )[:5]
    top_neg = sorted(
        [{"feature": k.replace("_", " ").title(), "value": round(v * 100, 2)}
         for k, v in contributions.items() if v < 0],
        key=lambda x: x["value"]
    )[:3]

    return PredictionResult(
        score=score, probability=round(prob, 4),
        verdict=verdict, advice=advice, level=level, emoji=emoji,
        features=features, contributions=contributions,
        top_positive=top_pos, top_negative=top_neg,
        insights=insights,
        used_ml_model=used_ml,
        model_name=get_best_model_name(),
    )


def batch_predict(pairs: list[dict]) -> list[dict]:
    """Predict for a list of input dicts (same keys as predict()). Max 50."""
    results = []
    for p in pairs[:50]:
        r = predict(**{k: p[k] for k in p if k in predict.__code__.co_varnames})
        results.append({"score": r.score, "verdict": r.verdict, "level": r.level})
    return results
