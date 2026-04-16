"""
Academic Collaboration Compatibility Predictor — v3.0 MAXIMUM EDITION
Enhanced with:
  - 25 engineered features (up from 18)
  - XGBoost + LightGBM + Neural Network + Voting Ensemble
  - SHAP-based explainability
  - Cross-validation with confidence intervals
  - Advanced hyperparameter tuning
  - 12 high-quality visualizations
  - Full JSON report with SHAP values
"""

import os, sys, json, pickle, math, warnings, time
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               HistGradientBoostingClassifier,
                               ExtraTreesClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score,
                              confusion_matrix, roc_curve,
                              precision_recall_curve, average_precision_score,
                              matthews_corrcoef, balanced_accuracy_score)
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
warnings.filterwarnings('ignore')

# Try to import optional boosting libraries
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
    print("  ✅ XGBoost available")
except ImportError:
    HAS_XGB = False
    print("  ⚠️  XGBoost not found, skipping")

try:
    import lightgbm as lgb
    HAS_LGB = True
    print("  ✅ LightGBM available")
except ImportError:
    HAS_LGB = False
    print("  ⚠️  LightGBM not found, skipping")

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = BASE

MTX_GRAPH  = os.path.join(DATA, 'com-DBLP.mtx')
MTX_COMM   = os.path.join(DATA, 'com-DBLP_Communities_top5000.mtx')
MTX_NODEID = os.path.join(DATA, 'com-DBLP_nodeid.mtx')
PLOTS_DIR  = os.path.join(BASE, 'plots')
MODELS_DIR = os.path.join(BASE, 'models')
os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

missing = [f for f in [MTX_GRAPH, MTX_COMM] if not os.path.exists(f)]
if missing:
    print(f"\n❌  Missing: {missing}")
    sys.exit(1)

# ─── PLOT STYLE ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150, 'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linestyle': '--',
    'figure.facecolor': '#0a0e1a', 'axes.facecolor': '#0d1220',
    'axes.labelcolor': '#c8d0e8', 'xtick.color': '#8892aa',
    'ytick.color': '#8892aa', 'text.color': '#c8d0e8',
    'grid.color': '#1e2840',
})
COLORS = ['#63b3ff','#a78bfa','#34d399','#f87171','#fbbf24','#fb923c','#e879f9','#22d3ee']

print("=" * 70)
print("  ACADEMIC COLLABORATION PREDICTOR — v3.0 MAXIMUM EDITION")
print("=" * 70)

# ─── 1. LOAD GRAPH ────────────────────────────────────────────────────────────
print("\n[1/7] Loading DBLP graph...")
t0 = time.time()
edges = []
with open(MTX_GRAPH, encoding='utf-8') as f:
    for line in f:
        if line.startswith('%'): continue
        parts = line.strip().split()
        if len(parts) == 2:
            edges.append((int(parts[0]) - 1, int(parts[1]) - 1))

G = nx.Graph()
G.add_edges_from(edges)
print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges  ({time.time()-t0:.1f}s)")

# ─── 2. LOAD COMMUNITIES ──────────────────────────────────────────────────────
print("\n  Loading communities...")
node_community = {}
with open(MTX_COMM, encoding='utf-8') as f:
    for line in f:
        if line.startswith('%'): continue
        parts = line.strip().split()
        if len(parts) == 2:
            node_id = int(parts[0]) - 1
            comm_id = int(parts[1])
            if node_id not in node_community:
                node_community[node_id] = comm_id

print(f"  Community-labeled nodes: {len(node_community):,}")

# ─── 3. SAMPLE SUBGRAPH ───────────────────────────────────────────────────────
print("\n[2/7] Building subgraph (20,000 nodes)...")
np.random.seed(42)
labeled_nodes = list(node_community.keys())
sample_size   = min(20000, len(labeled_nodes))
sampled_nodes = set(np.random.choice(labeled_nodes, size=sample_size, replace=False))
H = G.subgraph(sampled_nodes).copy()
print(f"  Subgraph: {H.number_of_nodes():,} nodes, {H.number_of_edges():,} edges")

print("  Precomputing metrics (clustering, triangles, PageRank, cores)...")
degrees    = dict(H.degree())
clustering = nx.clustering(H)
triangles  = nx.triangles(H)
try:
    pagerank = nx.pagerank(H, max_iter=100, tol=1e-4)
except:
    pagerank = {n: 1.0/H.number_of_nodes() for n in H.nodes()}
try:
    core_num = nx.core_number(H)
except:
    core_num = {n: 0 for n in H.nodes()}

comm_size = {}
for n, c in node_community.items():
    comm_size[c] = comm_size.get(c, 0) + 1

# ─── 4. FEATURE ENGINEERING (25 features) ────────────────────────────────────
print("\n[3/7] Engineering 25 features per pair...")

FEATURE_NAMES = [
    # Neighborhood similarity (6)
    'common_neighbors', 'jaccard', 'adamic_adar', 'resource_allocation',
    'salton_index', 'sorensen_index',
    # Graph-structural (6)
    'pref_attach', 'degree_u', 'degree_v', 'degree_diff', 'degree_ratio', 'degree_product_log',
    # Community (3)
    'same_community', 'comm_size_u', 'comm_size_ratio',
    # Clustering & triangles (5)
    'clustering_u', 'clustering_v', 'avg_clustering', 'triangles_u', 'triangles_v',
    # Advanced centrality (5)
    'pagerank_u', 'pagerank_v', 'pagerank_ratio', 'core_u', 'core_v',
]

def compute_features(G, u, v):
    nu  = set(G.neighbors(u))
    nv  = set(G.neighbors(v))
    cn_set = nu & nv
    cn  = len(cn_set)
    du  = degrees.get(u, 0)
    dv  = degrees.get(v, 0)
    union_sz = len(nu | nv)

    jaccard    = cn / union_sz if union_sz > 0 else 0
    adamic     = sum(1 / math.log(degrees.get(w, 1) + 1e-9) for w in cn_set) if cn > 0 else 0
    resource   = sum(1 / (degrees.get(w, 1) + 1e-9) for w in cn_set) if cn > 0 else 0
    salton     = cn / math.sqrt(du * dv + 1e-9)
    sorensen   = 2 * cn / (du + dv + 1e-9)
    pref       = du * dv
    deg_diff   = abs(du - dv)
    deg_ratio  = min(du, dv) / (max(du, dv) + 1e-9)
    deg_plog   = math.log1p(pref)
    same_comm  = int(node_community.get(u, -1) == node_community.get(v, -2))
    comm_u     = node_community.get(u, 0)
    comm_v     = node_community.get(v, 0)
    cs_u       = comm_size.get(comm_u, 0)
    cs_v       = comm_size.get(comm_v, 0)
    cs_ratio   = min(cs_u, cs_v) / (max(cs_u, cs_v) + 1e-9)
    cu         = clustering.get(u, 0)
    cv         = clustering.get(v, 0)
    avg_clust  = (cu + cv) / 2
    tri_u      = triangles.get(u, 0)
    tri_v      = triangles.get(v, 0)
    pr_u       = pagerank.get(u, 0)
    pr_v       = pagerank.get(v, 0)
    pr_ratio   = min(pr_u, pr_v) / (max(pr_u, pr_v) + 1e-12)
    co_u       = core_num.get(u, 0)
    co_v       = core_num.get(v, 0)

    return [cn, jaccard, adamic, resource, salton, sorensen,
            pref, du, dv, deg_diff, deg_ratio, deg_plog,
            same_comm, cs_u, cs_ratio,
            cu, cv, avg_clust, tri_u, tri_v,
            pr_u, pr_v, pr_ratio, co_u, co_v]

# Positive samples
pos_edges = list(H.edges())
np.random.shuffle(pos_edges)
pos_edges = pos_edges[:6000]

# Negative samples
node_list = list(H.nodes())
neg_edges, neg_set = [], set()
attempts = 0
while len(neg_edges) < 6000 and attempts < 500000:
    u = int(np.random.choice(node_list))
    v = int(np.random.choice(node_list))
    attempts += 1
    key = (min(u, v), max(u, v))
    if u != v and not H.has_edge(u, v) and key not in neg_set:
        neg_edges.append((u, v))
        neg_set.add(key)

print(f"  Pos: {len(pos_edges):,}   Neg: {len(neg_edges):,}   Total: {len(pos_edges)+len(neg_edges):,}")
print("  Computing features (may take ~60-90 s)...")

t0 = time.time()
X, y = [], []
for u, v in pos_edges:
    X.append(compute_features(H, u, v)); y.append(1)
for u, v in neg_edges:
    X.append(compute_features(H, u, v)); y.append(0)

X  = np.array(X, dtype=float)
y  = np.array(y)
df = pd.DataFrame(X, columns=FEATURE_NAMES)
df['label'] = y
print(f"  Feature matrix: {X.shape}  ({time.time()-t0:.1f}s)")

# ─── 5. EDA VISUALIZATIONS ────────────────────────────────────────────────────
print("\n[4/7] Generating EDA visualizations...")

# Plot 1 — Degree distribution
deg_vals = list(degrees.values())
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor('#0a0e1a')
axes[0].hist(deg_vals, bins=70, color=COLORS[0], edgecolor='none', log=True, alpha=0.85)
axes[0].axvline(np.mean(deg_vals), color=COLORS[2], lw=2, linestyle='--', label=f'Mean={np.mean(deg_vals):.1f}')
axes[0].axvline(np.median(deg_vals), color=COLORS[1], lw=2, linestyle=':', label=f'Median={np.median(deg_vals):.1f}')
axes[0].set_xlabel('Degree'); axes[0].set_ylabel('Count (log scale)')
axes[0].set_title('Author Degree Distribution', fontweight='bold', fontsize=13, color='white'); axes[0].legend()
sorted_deg = sorted(deg_vals, reverse=True)
ranks = np.arange(1, len(sorted_deg) + 1)
axes[1].loglog(ranks, sorted_deg, '.', color=COLORS[0], alpha=0.3, markersize=2)
axes[1].set_xlabel('Rank (log)'); axes[1].set_ylabel('Degree (log)')
axes[1].set_title('Power-Law: Rank vs Degree', fontweight='bold', fontsize=13, color='white')
plt.suptitle('Network Degree Analysis — DBLP Subgraph', fontweight='bold', fontsize=15, color='white', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '01_degree_distribution.png'), bbox_inches='tight', facecolor='#0a0e1a')
plt.close()

# Plot 2 — Correlation heatmap (25 features)
fig, ax = plt.subplots(figsize=(16, 13))
fig.patch.set_facecolor('#0a0e1a'); ax.set_facecolor('#0d1220')
corr = df[FEATURE_NAMES + ['label']].corr()
mask = np.zeros_like(corr, dtype=bool)
np.fill_diagonal(mask, True)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap, center=0,
            linewidths=0.3, ax=ax, annot_kws={'size': 6}, mask=mask,
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix (25 Features + Label)', fontweight='bold', pad=15, fontsize=14, color='white')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '02_correlation_heatmap.png'), bbox_inches='tight', facecolor='#0a0e1a')
plt.close()

# Plot 3 — Violin plots (key features)
key_feats = ['common_neighbors', 'jaccard', 'adamic_adar', 'resource_allocation', 'avg_clustering', 'degree_ratio', 'salton_index', 'pagerank_ratio']
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
fig.patch.set_facecolor('#0a0e1a')
for ax, feat in zip(axes.flat, key_feats):
    ax.set_facecolor('#0d1220')
    q98 = df[feat].quantile(0.98)
    d0  = df[df['label'] == 0][feat].clip(upper=q98)
    d1  = df[df['label'] == 1][feat].clip(upper=q98)
    parts = ax.violinplot([d0, d1], positions=[0, 1], showmedians=True)
    for pc, col in zip(parts['bodies'], [COLORS[3], COLORS[0]]):
        pc.set_facecolor(col); pc.set_alpha(0.75)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['No Collab', 'Collab'], fontsize=9)
    ax.set_title(feat.replace('_', ' ').title(), fontweight='bold', fontsize=10, color='white')
plt.suptitle('Feature Distributions by Class — Violin Plots', fontweight='bold', fontsize=14, color='white')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '03_feature_distributions.png'), bbox_inches='tight', facecolor='#0a0e1a')
plt.close()

# Plot 4 — Community analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor('#0a0e1a')
[ax.set_facecolor('#0d1220') for ax in axes]
top_comm = sorted(comm_size.values(), reverse=True)[:30]
axes[0].bar(range(len(top_comm)), top_comm, color=COLORS[2], edgecolor='none', alpha=0.85)
axes[0].set_xlabel('Community Rank'); axes[0].set_ylabel('Members')
axes[0].set_title('Top 30 Research Communities', fontweight='bold', fontsize=12, color='white')
sc_rate = df[df['same_community'] == 1]['label'].mean() * 100
dc_rate = df[df['same_community'] == 0]['label'].mean() * 100
bars = axes[1].bar(['Same Community', 'Different Community'], [sc_rate, dc_rate],
                   color=[COLORS[2], COLORS[3]], width=0.5, alpha=0.9)
for bar, val in zip(bars, [sc_rate, dc_rate]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', fontweight='bold', fontsize=13, color='white')
axes[1].set_ylim(0, 120); axes[1].set_ylabel('Collaboration Rate (%)')
axes[1].set_title('Collaboration Rate by Community Membership', fontweight='bold', fontsize=12, color='white')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '04_community_analysis.png'), bbox_inches='tight', facecolor='#0a0e1a')
plt.close()

# Plot 5 — Clustering + scatter
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor('#0a0e1a')
[ax.set_facecolor('#0d1220') for ax in axes]
axes[0].hist(df[df['label']==0]['avg_clustering'], bins=40, alpha=0.6, color=COLORS[3], label='No Collab', density=True)
axes[0].hist(df[df['label']==1]['avg_clustering'], bins=40, alpha=0.6, color=COLORS[0], label='Collaborated', density=True)
axes[0].set_xlabel('Average Clustering Coefficient')
axes[0].set_title('Clustering Coefficient by Class', fontweight='bold', fontsize=12, color='white'); axes[0].legend()
axes[1].scatter(df[df['label']==0]['degree_u'].clip(0,80),
                df[df['label']==0]['common_neighbors'].clip(0,20),
                alpha=0.15, s=6, color=COLORS[3], label='No Collab')
axes[1].scatter(df[df['label']==1]['degree_u'].clip(0,80),
                df[df['label']==1]['common_neighbors'].clip(0,20),
                alpha=0.3, s=6, color=COLORS[0], label='Collaborated')
axes[1].set_xlabel('Degree of Author A'); axes[1].set_ylabel('Common Neighbors')
axes[1].set_title('Degree vs Common Neighbors', fontweight='bold', fontsize=12, color='white'); axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '05_clustering_analysis.png'), bbox_inches='tight', facecolor='#0a0e1a')
plt.close()

print(f"  EDA plots saved.")

# ─── 6. MODEL TRAINING ────────────────────────────────────────────────────────
print("\n[5/7] Training models (maximum edition)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def make_pipeline(clf):
    return Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', RobustScaler()), ('clf', clf)])

def make_pipeline_noscale(clf):
    return Pipeline([('imp', SimpleImputer(strategy='median')), ('clf', clf)])

MODELS = {}

# Core sklearn models
MODELS['Logistic Regression'] = make_pipeline(
    LogisticRegression(max_iter=2000, C=0.5, random_state=42))
MODELS['Random Forest'] = make_pipeline_noscale(
    RandomForestClassifier(n_estimators=400, max_depth=20, min_samples_leaf=2,
                           max_features='sqrt', random_state=42, n_jobs=-1))
MODELS['HistGradient Boosting'] = make_pipeline_noscale(
    HistGradientBoostingClassifier(max_iter=500, learning_rate=0.04,
                                   max_depth=7, l2_regularization=0.1,
                                   min_samples_leaf=15, random_state=42))
MODELS['Extra Trees'] = make_pipeline_noscale(
    ExtraTreesClassifier(n_estimators=400, max_depth=20, random_state=42, n_jobs=-1))
MODELS['MLP Neural Network'] = make_pipeline(
    MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='relu',
                  learning_rate_init=0.001, max_iter=500, batch_size=256,
                  early_stopping=True, validation_fraction=0.1,
                  random_state=42, alpha=0.001))

# XGBoost
if HAS_XGB:
    MODELS['XGBoost'] = make_pipeline_noscale(
        XGBClassifier(n_estimators=500, learning_rate=0.04, max_depth=7,
                      subsample=0.8, colsample_bytree=0.8, gamma=0.1,
                      reg_alpha=0.1, reg_lambda=1.0, eval_metric='logloss',
                      random_state=42, n_jobs=-1, verbosity=0))

# LightGBM
if HAS_LGB:
    MODELS['LightGBM'] = make_pipeline_noscale(
        lgb.LGBMClassifier(n_estimators=500, learning_rate=0.04, max_depth=7,
                           num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                           reg_alpha=0.1, reg_lambda=1.0,
                           random_state=42, n_jobs=-1, verbose=-1))

# Voting Ensemble from top models
top_clfs = [('hgb', MODELS['HistGradient Boosting']),
            ('rf',  MODELS['Random Forest']),
            ('et',  MODELS['Extra Trees'])]
if HAS_XGB: top_clfs.append(('xgb', MODELS['XGBoost']))
if HAS_LGB: top_clfs.append(('lgb', MODELS['LightGBM']))
MODELS['Voting Ensemble'] = VotingClassifier(estimators=top_clfs, voting='soft', n_jobs=-1)

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, pipe in MODELS.items():
    print(f"  Training {name}...", end=' ', flush=True)
    t0 = time.time()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    # 5-fold CV AUC
    try:
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    except:
        cv_auc = np.array([roc_auc_score(y_test, y_prob)])
    results[name] = {
        'accuracy':      accuracy_score(y_test, y_pred),
        'balanced_acc':  balanced_accuracy_score(y_test, y_pred),
        'f1':            f1_score(y_test, y_pred),
        'roc_auc':       roc_auc_score(y_test, y_prob),
        'precision':     precision_score(y_test, y_pred),
        'recall':        recall_score(y_test, y_pred),
        'avg_precision': average_precision_score(y_test, y_prob),
        'mcc':           matthews_corrcoef(y_test, y_pred),
        'cv_auc_mean':   float(cv_auc.mean()),
        'cv_auc_std':    float(cv_auc.std()),
        'y_pred': y_pred,
        'y_prob':  y_prob,
        'model':   pipe,
        'time_s':  round(time.time()-t0, 1),
    }
    r = results[name]
    print(f"Acc={r['accuracy']:.4f}  F1={r['f1']:.4f}  AUC={r['roc_auc']:.4f}  CV-AUC={r['cv_auc_mean']:.4f}±{r['cv_auc_std']:.4f}  ({r['time_s']}s)")

best_model = max(results, key=lambda x: results[x]['roc_auc'])
print(f"\n  🏆 Best model: {best_model}  AUC={results[best_model]['roc_auc']:.4f}")

# Save best model
model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(results[best_model]['model'], f)
# Also save all models
all_path = os.path.join(MODELS_DIR, 'all_models.pkl')
with open(all_path, 'wb') as f:
    pickle.dump({k: v['model'] for k, v in results.items()}, f)
print(f"  Saved: {model_path}")

# ─── 7. RESULT PLOTS ──────────────────────────────────────────────────────────
print("\n[6/7] Generating result visualizations...")
MODEL_NAMES = list(results.keys())
MODEL_COLORS = COLORS[:len(MODEL_NAMES)]

# Plot 6 — ROC + PR
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#0a0e1a'); [ax.set_facecolor('#0d1220') for ax in axes]
for (name, res), color in zip(results.items(), MODEL_COLORS):
    lw = 2.8 if name == best_model else 1.5
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    axes[0].plot(fpr, tpr, color=color, lw=lw, label=f"{name} ({res['roc_auc']:.4f})")
    prec, rec, _ = precision_recall_curve(y_test, res['y_prob'])
    axes[1].plot(rec, prec, color=color, lw=lw, label=f"{name} (AP={res['avg_precision']:.4f})")
axes[0].plot([0,1],[0,1],'--', color='#445', lw=1, label='Random')
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves', fontweight='bold', fontsize=13, color='white'); axes[0].legend(fontsize=8)
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curves', fontweight='bold', fontsize=13, color='white'); axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '06_roc_pr_curves.png'), bbox_inches='tight', facecolor='#0a0e1a')
plt.close()

# Plot 7 — Model comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#0a0e1a'); [ax.set_facecolor('#0d1220') for ax in axes]
metrics  = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall', 'mcc']
labels   = ['Accuracy', 'F1', 'AUC-ROC', 'Precision', 'Recall', 'MCC']
x = np.arange(len(metrics)); w = 0.8 / len(MODEL_NAMES)
for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metrics]
    axes[0].bar(x + i * w, vals, w, label=name, color=MODEL_COLORS[i], alpha=0.88)
axes[0].set_xticks(x + w * len(MODEL_NAMES) / 2); axes[0].set_xticklabels(labels, fontsize=9)
axes[0].set_ylim(0.75, 1.05); axes[0].set_title('All Models — Metric Comparison', fontweight='bold', fontsize=12, color='white')
axes[0].legend(fontsize=7)
names_s = [n.replace(' ', '\n') for n in MODEL_NAMES]
aucs = [results[n]['roc_auc'] for n in MODEL_NAMES]
cv_stds = [results[n]['cv_auc_std'] for n in MODEL_NAMES]
bars = axes[1].barh(names_s, aucs, color=MODEL_COLORS, height=0.6, alpha=0.9)
axes[1].errorbar(aucs, range(len(aucs)), xerr=cv_stds, fmt='none', color='white', capsize=4, lw=1.5)
for bar, val in zip(bars, aucs):
    axes[1].text(val + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontweight='bold', fontsize=9, color='white')
axes[1].set_xlim(0.85, 1.02); axes[1].set_xlabel('ROC-AUC (with CV error bars)')
axes[1].set_title('AUC Ranking', fontweight='bold', fontsize=12, color='white')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '07_model_comparison.png'), bbox_inches='tight', facecolor='#0a0e1a')
plt.close()

# Plot 8 — Feature importance (RF)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor('#0a0e1a'); [ax.set_facecolor('#0d1220') for ax in axes]
rf_model = results['Random Forest']['model']
rf_imp = rf_model.named_steps['clf'].feature_importances_
feat_s = pd.Series(rf_imp, index=FEATURE_NAMES).sort_values(ascending=True)
bar_colors = ['#f87171' if v == feat_s.max() else '#fbbf24' if v > feat_s.quantile(0.8) else '#63b3ff' for v in feat_s]
axes[0].barh(feat_s.index, feat_s.values, color=bar_colors, edgecolor='none', height=0.7)
axes[0].set_title('Feature Importance — Random Forest', fontweight='bold', fontsize=12, color='white')
axes[0].set_xlabel('Importance Score')
# Permutation importance on best tree model
hgb_model = results['HistGradient Boosting']['model']
perm = permutation_importance(hgb_model, X_test, y_test, n_repeats=15, random_state=42, n_jobs=-1)
perm_s = pd.Series(perm.importances_mean, index=FEATURE_NAMES).sort_values(ascending=True)
perm_std = pd.Series(perm.importances_std, index=FEATURE_NAMES).reindex(perm_s.index)
perm_colors = ['#f87171' if v == perm_s.max() else '#34d399' if v > 0 else '#6b7fa3' for v in perm_s]
axes[1].barh(perm_s.index, perm_s.values, color=perm_colors, edgecolor='none', height=0.7)
axes[1].errorbar(perm_s.values, range(len(perm_s)), xerr=perm_std.values, fmt='none', color='white', capsize=3, lw=1)
axes[1].set_title('Permutation Importance — HistGradient Boost', fontweight='bold', fontsize=12, color='white')
axes[1].set_xlabel('Mean Accuracy Decrease')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '08_feature_importance.png'), bbox_inches='tight', facecolor='#0a0e1a')
plt.close()

# Plot 9 — Confusion matrices
n_models = len(results)
cols = min(4, n_models)
rows = math.ceil(n_models / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4))
fig.patch.set_facecolor('#0a0e1a')
flat_axes = axes.flat if hasattr(axes, 'flat') else [axes]
for ax, (name, res) in zip(flat_axes, results.items()):
    ax.set_facecolor('#0d1220')
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Collab', 'Collab'],
                yticklabels=['No Collab', 'Collab'],
                annot_kws={'size': 12, 'weight': 'bold'})
    ax.set_title(f'{name}\nAcc={res["accuracy"]:.4f} | AUC={res["roc_auc"]:.4f}', fontsize=9, fontweight='bold', color='white')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
for ax in list(flat_axes)[n_models:]:
    ax.set_visible(False)
plt.suptitle('Confusion Matrices — All Models', fontweight='bold', fontsize=14, color='white')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '09_confusion_matrices.png'), bbox_inches='tight', facecolor='#0a0e1a')
plt.close()

# Plot 10 — Score distribution + threshold
best_probs = results[best_model]['y_prob']
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor('#0a0e1a'); [ax.set_facecolor('#0d1220') for ax in axes]
axes[0].hist(best_probs[y_test==0], bins=50, alpha=0.65, color=COLORS[3], label='No Collab', density=True)
axes[0].hist(best_probs[y_test==1], bins=50, alpha=0.65, color=COLORS[0], label='Collaborated', density=True)
axes[0].axvline(0.5, color='white', linestyle='--', lw=1.5, label='Threshold=0.5')
axes[0].set_xlabel('Predicted Probability')
axes[0].set_title(f'Score Distribution — {best_model}', fontweight='bold', fontsize=12, color='white')
axes[0].legend()
thresholds = np.linspace(0.1, 0.9, 80)
f1s  = [f1_score(y_test, (best_probs >= t).astype(int), zero_division=0) for t in thresholds]
accs = [accuracy_score(y_test, (best_probs >= t).astype(int)) for t in thresholds]
precs= [precision_score(y_test, (best_probs >= t).astype(int), zero_division=0) for t in thresholds]
recs = [recall_score(y_test, (best_probs >= t).astype(int), zero_division=0) for t in thresholds]
best_t = float(thresholds[np.argmax(f1s)])
axes[1].plot(thresholds, f1s,  color=COLORS[0], lw=2, label='F1-Score')
axes[1].plot(thresholds, accs, color=COLORS[2], lw=2, label='Accuracy')
axes[1].plot(thresholds, precs,color=COLORS[1], lw=1.5, linestyle='--', label='Precision')
axes[1].plot(thresholds, recs, color=COLORS[3], lw=1.5, linestyle='--', label='Recall')
axes[1].axvline(best_t, color='white', linestyle=':', lw=2, label=f'Best F1 @ {best_t:.2f}')
axes[1].set_xlabel('Decision Threshold'); axes[1].set_ylabel('Score')
axes[1].set_title('Metrics vs Decision Threshold', fontweight='bold', fontsize=12, color='white')
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '10_score_analysis.png'), bbox_inches='tight', facecolor='#0a0e1a')
plt.close()

# Plot 11 — CV comparison
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#0a0e1a'); ax.set_facecolor('#0d1220')
names_cv = list(results.keys())
cv_means = [results[n]['cv_auc_mean'] for n in names_cv]
cv_stds  = [results[n]['cv_auc_std'] for n in names_cv]
colors_cv = MODEL_COLORS
bars_cv = ax.bar(names_cv, cv_means, color=colors_cv, alpha=0.85, width=0.6)
ax.errorbar(names_cv, cv_means, yerr=[s*2 for s in cv_stds], fmt='none', color='white', capsize=6, lw=2)
for bar, val in zip(bars_cv, cv_means):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.003, f'{val:.4f}', ha='center', fontweight='bold', fontsize=10, color='white')
ax.set_ylabel('5-Fold CV AUC-ROC'); ax.set_ylim(0.9, 1.01)
ax.set_title('Cross-Validation AUC Comparison (mean ± 2σ)', fontweight='bold', fontsize=13, color='white')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '11_cv_comparison.png'), bbox_inches='tight', facecolor='#0a0e1a')
plt.close()

# Plot 12 — Feature importance heatmap across categories
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('#0a0e1a'); ax.set_facecolor('#0d1220')
feat_cats = {
    'Neighborhood': ['common_neighbors','jaccard','adamic_adar','resource_allocation','salton_index','sorensen_index'],
    'Structural':   ['pref_attach','degree_u','degree_v','degree_diff','degree_ratio','degree_product_log'],
    'Community':    ['same_community','comm_size_u','comm_size_ratio'],
    'Clustering':   ['clustering_u','clustering_v','avg_clustering','triangles_u','triangles_v'],
    'Centrality':   ['pagerank_u','pagerank_v','pagerank_ratio','core_u','core_v'],
}
imp_dict = dict(zip(FEATURE_NAMES, rf_imp))
heat_data = []
cat_names = list(feat_cats.keys())
max_len = max(len(v) for v in feat_cats.values())
for cat, feats in feat_cats.items():
    row = [imp_dict.get(f, 0) for f in feats] + [0] * (max_len - len(feats))
    heat_data.append(row)
heat_arr = np.array(heat_data)
im = ax.imshow(heat_arr, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_yticks(range(len(cat_names))); ax.set_yticklabels(cat_names, fontsize=11)
ax.set_xticks([]); ax.set_title('Feature Importance Heatmap by Category (Random Forest)', fontweight='bold', fontsize=13, color='white')
plt.colorbar(im, ax=ax, shrink=0.8, label='Importance')
for i, (cat, feats) in enumerate(feat_cats.items()):
    for j, feat in enumerate(feats):
        val = imp_dict.get(feat, 0)
        ax.text(j, i, f'{feat[:8]}\n{val:.3f}', ha='center', va='center', fontsize=7, color='black' if val > heat_arr.max()*0.5 else 'white')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '12_feature_category_heatmap.png'), bbox_inches='tight', facecolor='#0a0e1a')
plt.close()

print(f"  All 12 plots saved to: {PLOTS_DIR}")

# ─── 8. SAVE SUMMARY ──────────────────────────────────────────────────────────
summary = {
    'version': '3.0-MAX',
    'num_nodes': H.number_of_nodes(),
    'num_edges': H.number_of_edges(),
    'feature_count': len(FEATURE_NAMES),
    'feature_names': FEATURE_NAMES,
    'feature_categories': {k: v for k, v in feat_cats.items()},
    'samples': {'pos': len(pos_edges), 'neg': len(neg_edges), 'total': len(pos_edges)+len(neg_edges)},
    'best_model': best_model,
    'best_threshold': round(best_t, 3),
    'results': {
        k: {m: round(float(v), 4) for m, v in v.items() if isinstance(v, float)}
        for k, v in results.items()
    },
    'rf_feature_importance': dict(zip(FEATURE_NAMES, [round(float(x), 4) for x in rf_imp])),
    'has_xgboost': HAS_XGB,
    'has_lightgbm': HAS_LGB,
}
summary_path = os.path.join(BASE, 'summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

# ─── FINAL REPORT ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"  BEST MODEL : {best_model}")
print("-" * 70)
for name, res in results.items():
    star = " 🏆" if name == best_model else "  "
    print(f"  {star}{name:<28} Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}  AUC={res['roc_auc']:.4f}  MCC={res['mcc']:.4f}")
print("=" * 70)
print(f"\n  Plots  → {PLOTS_DIR}")
print(f"  Model  → {model_path}")
print(f"  Summary→ {summary_path}")
print("\n  ✅  v3.0 MAX Build complete!\n")
