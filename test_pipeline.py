from pipeline import predict, batch_predict, get_feature_importance, get_summary, get_best_model_name

# ═══════════════════════════════════════════════════════════════
# Single Prediction
# ═══════════════════════════════════════════════════════════════
print("🧪 SINGLE PREDICTION TEST")
print("─" * 50)

result = predict(
    cn=8,                    # Common neighbors
    du=25,                   # Degree of user 1
    dv=30,                   # Degree of user 2
    same_community=1,        # 1 if same, 0 if different
    clust_u=0.15,            # Clustering coefficient user 1
    clust_v=0.12,            # Clustering coefficient user 2
    tri_u=5,                 # Triangles for user 1
    tri_v=3,                 # Triangles for user 2
    cs_u=150,                # Community size user 1
    cs_v=200,                # Community size user 2
    pr_u=0.0005,             # PageRank user 1
    pr_v=0.0003,             # PageRank user 2
    core_u=3,                # K-core user 1
    core_v=2,                # K-core user 2
)

print(f"✓ Score: {result.score}")
print(f"✓ Verdict: {result.verdict}")
print(f"✓ Probability: {result.probability}")
print(f"✓ Level: {result.level}")
print(f"✓ Advice: {result.advice}\n")

# ═══════════════════════════════════════════════════════════════
# Batch Prediction (Multiple Pairs)
# ═════════════════════════════════════════════════════════════
print("📊 BATCH PREDICTION TEST")
print("─" * 50)

pairs = [
    {"cn": 5, "du": 20, "dv": 15, "same_community": 1, "clust_u": 0.1, "clust_v": 0.1, "tri_u": 2, "tri_v": 2, "cs_u": 100, "cs_v": 100, "pr_u": 0.0001, "pr_v": 0.0001, "core_u": 2, "core_v": 2},
    {"cn": 10, "du": 40, "dv": 35, "same_community": 0, "clust_u": 0.2, "clust_v": 0.15, "tri_u": 8, "tri_v": 6, "cs_u": 300, "cs_v": 250, "pr_u": 0.0008, "pr_v": 0.0007, "core_u": 4, "core_v": 3},
]

batch_results = batch_predict(pairs)
for i, res in enumerate(batch_results, 1):
    print(f"  Pair {i}: {res['verdict']} (Score: {res['score']}, Level: {res['level']})")

# ═══════════════════════════════════════════════════════════════
# Model Info
# ═════════════════════════════════════════════════════════════
print("\n📈 MODEL INFORMATION")
print("─" * 50)
print(f"✓ Best Model: {get_best_model_name()}")
print(f"✓ Summary: {get_summary()}")

# ═════════════════════════════════════════════════════════════
# Feature Importance
# ═════════════════════════════════════════════════════════════
print(f"\n🎯 TOP 5 IMPORTANT FEATURES")
print("─" * 50)
importance = get_feature_importance()
top_5 = sorted(importance.items(), key=lambda x: -x[1])[:5]
for feat, imp in top_5:
    print(f"  {feat:25} {imp:.4f}")