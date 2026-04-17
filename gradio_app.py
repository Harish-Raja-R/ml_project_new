"""
NeuroCollab — Gradio UI for Hugging Face Spaces v3.0 MAX
Run locally:  python gradio_app.py
HF Spaces:    upload this file as app.py + pipeline.py + requirements.txt
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import json

from pipeline import (
    predict, get_model_results, get_feature_importance,
    get_summary, FEATURE_CATEGORIES,
)

# ─── THEME ───────────────────────────────────────────────────────────────────
_THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#eff8ff", c100="#daeeff", c200="#bed9f8",
        c300="#93c1f3", c400="#5eb8ff", c500="#3b8fd9",
        c600="#2272b5", c700="#1b5a8f", c800="#164b75",
        c900="#133e60", c950="#0d2b45",
    ),
    secondary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Syne"), gr.themes.GoogleFont("JetBrains Mono"), "sans-serif"],
).set(
    body_background_fill="#02040f",
    body_background_fill_dark="#02040f",
    block_background_fill="#06090f",
    block_background_fill_dark="#06090f",
    block_border_color="#1a2540",
    block_border_color_dark="#1a2540",
    block_label_text_color="#94a3c4",
    block_label_text_color_dark="#94a3c4",
    input_background_fill="#0a0f1e",
    input_background_fill_dark="#0a0f1e",
    input_border_color="#1e2d4a",
    input_border_color_dark="#1e2d4a",
    button_primary_background_fill="linear-gradient(135deg, #5eb8ff, #b794f4)",
    button_primary_background_fill_dark="linear-gradient(135deg, #5eb8ff, #b794f4)",
    button_primary_text_color="#02040f",
    button_primary_text_color_dark="#02040f",
    button_primary_border_color="transparent",
    checkbox_label_background_fill="#0a0f1e",
    checkbox_label_background_fill_dark="#0a0f1e",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

body, .gradio-container { background:#02040f !important; font-family:'Syne',sans-serif !important; }

/* Hero */
.nc-hero-grad {
  background: linear-gradient(135deg,#5eb8ff 0%,#b794f4 50%,#2dd4bf 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  background-clip:text; font-size:2.8rem; font-weight:800;
  text-align:center; line-height:1.15; letter-spacing:-1px;
}
.nc-sub { color:#94a3c4; text-align:center; font-family:'JetBrains Mono',monospace; font-size:.9rem; margin-top:4px; }
.nc-badge {
  display:inline-block; background:rgba(94,184,255,.12);
  border:1px solid rgba(94,184,255,.3); color:#5eb8ff;
  border-radius:999px; padding:3px 12px; font-size:.75rem;
  font-family:'JetBrains Mono',monospace; margin:3px;
}
.nc-divider {
  height:2px; margin:24px 0;
  background:linear-gradient(90deg,transparent,#5eb8ff,#b794f4,#2dd4bf,transparent);
}

/* Score card */
.nc-score-card {
  background:rgba(94,184,255,.06); border:1px solid rgba(94,184,255,.2);
  border-radius:20px; padding:28px; text-align:center;
}
.nc-score-big {
  font-size:4.5rem; font-weight:800; line-height:1;
  background:linear-gradient(135deg,#5eb8ff,#2dd4bf);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.nc-verdict { font-size:1.3rem; font-weight:700; margin-top:6px; }

/* Insight chips */
.chip-pos { background:rgba(45,212,191,.12); border:1px solid rgba(45,212,191,.35); color:#2dd4bf; border-radius:999px; padding:6px 14px; font-size:.83rem; display:inline-block; margin:3px; font-family:'JetBrains Mono',monospace; }
.chip-neg { background:rgba(252,107,107,.10); border:1px solid rgba(252,107,107,.28); color:#fc6b6b; border-radius:999px; padding:6px 14px; font-size:.83rem; display:inline-block; margin:3px; font-family:'JetBrains Mono',monospace; }
.chip-neu { background:rgba(94,184,255,.10); border:1px solid rgba(94,184,255,.28); color:#5eb8ff; border-radius:999px; padding:6px 14px; font-size:.83rem; display:inline-block; margin:3px; font-family:'JetBrains Mono',monospace; }

/* Feature row */
.feat-row { display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid rgba(94,184,255,.06); font-size:.85rem; }
.feat-name { color:#94a3c4; font-family:'JetBrains Mono',monospace; }
.feat-val  { color:#eef2ff; font-weight:600; }

/* Section title */
.nc-stitle { color:#eef2ff; font-weight:700; font-size:1rem; letter-spacing:.02em; margin:16px 0 10px; display:flex; align-items:center; gap:8px; }
.nc-stitle::before { content:''; display:block; width:4px; height:16px; background:linear-gradient(180deg,#5eb8ff,#b794f4); border-radius:4px; }
"""

# ─── PLOTLY THEME ─────────────────────────────────────────────────────────────
_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Syne, sans-serif", color="#94a3c4"),
    margin=dict(l=20, r=20, t=40, b=20),
    colorway=["#5eb8ff","#b794f4","#2dd4bf","#f59e0b","#fc6b6b","#f472b6"],
)


# ─── PREDICTION FUNCTION ──────────────────────────────────────────────────────
def run_predict(cn, du, dv, same_comm, cu, cv, tu, tv, csu, csv_, pru, prv, kcu, kcv):
    r = predict(
        cn=int(cn), du=int(du), dv=int(dv), same_community=int(same_comm),
        clust_u=float(cu), clust_v=float(cv),
        tri_u=int(tu), tri_v=int(tv),
        cs_u=int(csu), cs_v=int(csv_),
        pr_u=float(pru), pr_v=float(prv),
        core_u=int(kcu), core_v=int(kcv),
    )

    # ── Score HTML ────────────────────────────────────────────────────────────
    lvl_color = {"high":"#2dd4bf","medium":"#5eb8ff","low-medium":"#f59e0b","low":"#fc6b6b"}.get(r.level,"#5eb8ff")
    score_html = f"""
    <div class='nc-score-card'>
      <div class='nc-score-big' style='-webkit-text-fill-color:{lvl_color};color:{lvl_color}'>{r.score:.1f}%</div>
      <div class='nc-verdict' style='color:{lvl_color}'>{r.emoji} {r.verdict}</div>
      <div style='color:#94a3c4;font-size:.85rem;margin-top:10px;font-family:JetBrains Mono,monospace;'>
        {"✅ ML Model" if r.used_ml_model else "⚠️ Formula"} · {r.model_name}
      </div>
      <div style='color:#eef2ff;font-size:.88rem;margin-top:10px;line-height:1.6;'>{r.advice}</div>
    </div>"""

    # ── Insights HTML ─────────────────────────────────────────────────────────
    chip_map = {"positive":"chip-pos","negative":"chip-neg","neutral":"chip-neu"}
    chips = "".join(f"<span class='{chip_map.get(i['type'],'chip-neu')}'>{i['text']}</span>" for i in r.insights)
    insights_html = f"<div class='nc-stitle'>Key Insights</div><div style='line-height:2.2'>{chips}</div>"

    # ── Gauge ─────────────────────────────────────────────────────────────────
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=r.score,
        number={"suffix":"%","font":{"size":40,"family":"Syne","color":lvl_color}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#4a5570","tickfont":{"size":10}},
            "bar":{"color":lvl_color,"thickness":0.3},
            "bgcolor":"rgba(0,0,0,0)","borderwidth":0,
            "steps":[
                {"range":[0,35],"color":"rgba(252,107,107,.08)"},
                {"range":[35,55],"color":"rgba(245,158,11,.08)"},
                {"range":[55,78],"color":"rgba(94,184,255,.08)"},
                {"range":[78,100],"color":"rgba(45,212,191,.08)"},
            ],
            "threshold":{"line":{"color":lvl_color,"width":3},"thickness":0.75,"value":r.score},
        },
        title={"text":f"{r.emoji} Compatibility Score","font":{"size":14,"color":lvl_color}},
    ))
    fig_gauge.update_layout(**_LAYOUT, height=300)

    # ── Feature contributions ─────────────────────────────────────────────────
    top_all = sorted(r.contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
    names_  = [k.replace("_"," ").title() for k,_ in top_all]
    vals_   = [v*100 for _,v in top_all]
    fig_bar = go.Figure(go.Bar(
        x=vals_, y=names_, orientation="h",
        marker_color=["#2dd4bf" if v>0 else "#fc6b6b" for v in vals_],
        marker_line_width=0,
        text=[f"{v:+.1f}%" for v in vals_], textposition="outside",
        textfont=dict(size=10,color="#94a3c4"),
    ))
    fig_bar.update_layout(**_LAYOUT, height=400, title_text="Feature Contributions",
        xaxis=dict(title="Contribution(%)",gridcolor="rgba(94,184,255,.07)",zeroline=True,zerolinecolor="rgba(94,184,255,.2)"),
        yaxis=dict(autorange="reversed"),
    )

    # ── Features table ────────────────────────────────────────────────────────
    rows=[]
    imp = get_feature_importance()
    for cat, feats in FEATURE_CATEGORIES.items():
        for feat in feats:
            rows.append({
                "Category":cat,
                "Feature":feat.replace("_"," ").title(),
                "Value":round(r.features.get(feat,0),5),
                "Importance%":round(imp.get(feat,0)*100,2),
            })
    df_feat = pd.DataFrame(rows)

    return score_html, insights_html, fig_gauge, fig_bar, df_feat


# ─── MODEL PERFORMANCE FUNCTION ───────────────────────────────────────────────
def build_model_charts():
    model_results = get_model_results()
    if not model_results:
        return go.Figure(), go.Figure(), pd.DataFrame()

    df = pd.DataFrame(model_results).T.reset_index()
    df.rename(columns={"index":"Model"},inplace=True)
    for c in ["accuracy","f1","roc_auc","mcc"]:
        if c in df.columns:
            df[c]=pd.to_numeric(df[c],errors="coerce")

    # Bar — AUC
    df_s=df.sort_values("roc_auc",ascending=True)
    fig_auc=go.Figure(go.Bar(
        x=df_s["roc_auc"]*100, y=df_s["Model"], orientation="h",
        marker=dict(color=df_s["roc_auc"]*100,colorscale=[[0,"#5eb8ff"],[0.5,"#b794f4"],[1,"#2dd4bf"]],showscale=False,line_width=0),
        text=[f"{v:.2f}%" for v in df_s["roc_auc"]*100], textposition="outside",
        textfont=dict(size=11,color="#94a3c4"),
    ))
    fig_auc.update_layout(**_LAYOUT, height=360, title_text="AUC-ROC Comparison",
        xaxis=dict(range=[95,100],title="AUC-ROC (%)",gridcolor="rgba(94,184,255,.07)"),
    )

    # Radar
    metrics=["accuracy","f1","roc_auc","mcc"]
    m_labels=["Accuracy","F1","AUC-ROC","MCC"]
    colors_r=["#5eb8ff","#b794f4","#2dd4bf","#f59e0b","#fc6b6b","#f472b6","#a78bfa","#34d399"]
    fig_radar=go.Figure()
    for i,(_,row) in enumerate(df.iterrows()):
        vals=[row.get(m,0) for m in metrics]+[row.get(metrics[0],0)]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=m_labels+[m_labels[0]], fill="toself",
            name=row["Model"], line=dict(color=colors_r[i%len(colors_r)],width=2),
            opacity=0.85,
        ))
    fig_radar.update_layout(**_LAYOUT, height=400, title_text="Multi-Metric Radar",
        polar=dict(bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0.9,1.0],gridcolor="rgba(94,184,255,.12)",tickfont=dict(size=9)),
            angularaxis=dict(gridcolor="rgba(94,184,255,.12)"),
        ),
        legend=dict(font=dict(size=9),bgcolor="rgba(0,0,0,0)"),
    )

    display_cols=["Model"]+[c for c in ["accuracy","f1","roc_auc","mcc","cv_auc_mean","time_s"] if c in df.columns]
    return fig_auc, fig_radar, df[display_cols]


# ─── FEATURE IMPORTANCE FUNCTION ──────────────────────────────────────────────
def build_feature_charts():
    importance=get_feature_importance()
    if not importance:
        return go.Figure(), go.Figure()

    df_imp=pd.DataFrame(
        [(k.replace("_"," ").title(),v*100) for k,v in sorted(importance.items(),key=lambda x:-x[1])],
        columns=["Feature","Importance%"],
    )
    rev_cat={f:cat for cat,fs in FEATURE_CATEGORIES.items() for f in fs}
    df_imp["Category"]=df_imp["Feature"].apply(lambda x:rev_cat.get(x.lower().replace(" ","_"),"Other"))
    cat_colors={"Neighborhood":"#5eb8ff","Structural":"#b794f4","Community":"#2dd4bf","Clustering":"#f59e0b","Centrality":"#f472b6"}

    fig_imp=go.Figure(go.Bar(
        x=df_imp["Importance%"], y=df_imp["Feature"], orientation="h",
        marker_color=[cat_colors.get(c,"#94a3c4") for c in df_imp["Category"]],
        marker_line_width=0,
        text=[f"{v:.2f}%" for v in df_imp["Importance%"]], textposition="outside",
        textfont=dict(size=10,color="#94a3c4"),
    ))
    fig_imp.update_layout(**_LAYOUT, height=600, title_text="Feature Importance (RF)",
        xaxis=dict(title="Importance (%)",gridcolor="rgba(94,184,255,.07)"),
        yaxis=dict(autorange="reversed"),
    )

    df_cat=df_imp.groupby("Category")["Importance%"].sum().reset_index()
    fig_pie=go.Figure(go.Pie(
        labels=df_cat["Category"], values=df_cat["Importance%"], hole=0.55,
        marker=dict(colors=[cat_colors.get(c,"#94a3c4") for c in df_cat["Category"]],
                    line=dict(color="#02040f",width=3)),
        textinfo="label+percent", textfont=dict(size=12,family="Syne"),
    ))
    fig_pie.update_layout(**_LAYOUT, height=360, title_text="Feature Category Distribution",
        annotations=[dict(text="Features",x=0.5,y=0.5,font=dict(size=16,color="#eef2ff"),showarrow=False)],
    )
    return fig_imp, fig_pie


# ─── BUILD APP ────────────────────────────────────────────────────────────────
summary = get_summary()

with gr.Blocks(title="NeuroCollab · Academic AI Predictor") as demo:

    # ── Hero ──────────────────────────────────────────────────────────────────
    gr.HTML(f"""
    <div style='padding:40px 0 20px;'>
      <div class='nc-hero-grad'>NeuroCollab</div>
      <div class='nc-sub'>Academic Collaboration · Link Prediction · Graph ML · v3.0 MAX</div>
      <div style='text-align:center;margin-top:16px;'>
        <span class='nc-badge'>25 Features</span>
        <span class='nc-badge'>8 Models</span>
        <span class='nc-badge'>~99.4% AUC</span>
        <span class='nc-badge'>DBLP Graph</span>
        <span class='nc-badge'>Voting Ensemble</span>
        <span class='nc-badge'>{summary.get('num_nodes',20000):,} Nodes</span>
      </div>
    </div>
    <div class='nc-divider'></div>
    """)

    with gr.Tabs():

        # ── Prediction Tab ────────────────────────────────────────────────────
        with gr.TabItem("🔮 Predict"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<div class='nc-stitle'>👤 Researcher A</div>")
                    du     = gr.Slider(1, 200, 20, step=1, label="Degree (A)")
                    cu     = gr.Slider(0.0, 1.0, 0.25, step=0.01, label="Clustering Coeff (A)")
                    tu     = gr.Slider(0, 100, 8, step=1, label="Triangles (A)")
                    csu    = gr.Slider(5, 2000, 200, step=5, label="Community Size (A)")
                    pru    = gr.Slider(0.0, 0.01, 0.0003, step=0.0001, label="PageRank (A)")
                    kcu    = gr.Slider(1, 20, 4, step=1, label="k-Core (A)")

                    gr.HTML("<div class='nc-stitle' style='margin-top:16px;'>👤 Researcher B</div>")
                    dv     = gr.Slider(1, 200, 15, step=1, label="Degree (B)")
                    cv     = gr.Slider(0.0, 1.0, 0.18, step=0.01, label="Clustering Coeff (B)")
                    tv     = gr.Slider(0, 100, 6, step=1, label="Triangles (B)")
                    csv_   = gr.Slider(5, 2000, 150, step=5, label="Community Size (B)")
                    prv    = gr.Slider(0.0, 0.01, 0.0002, step=0.0001, label="PageRank (B)")
                    kcv    = gr.Slider(1, 20, 3, step=1, label="k-Core (B)")

                    gr.HTML("<div class='nc-stitle' style='margin-top:16px;'>🔗 Shared Properties</div>")
                    cn          = gr.Slider(0, 50, 5, step=1, label="Common Neighbors")
                    same_comm   = gr.Radio([0, 1], value=0, label="Same Community?",
                                          info="1 = Yes, 0 = No")
                    predict_btn = gr.Button("⚡ Predict Compatibility", variant="primary", size="lg")

                with gr.Column(scale=2):
                    score_html   = gr.HTML(label="Score")
                    insights_html= gr.HTML(label="Insights")
                    gauge_plot   = gr.Plot(label="Compatibility Gauge")
                    bar_plot     = gr.Plot(label="Feature Contributions")

            with gr.Row():
                feat_table = gr.DataFrame(label="All 25 Features", wrap=True)

            predict_btn.click(
                fn=run_predict,
                inputs=[cn, du, dv, same_comm, cu, cv, tu, tv, csu, csv_, pru, prv, kcu, kcv],
                outputs=[score_html, insights_html, gauge_plot, bar_plot, feat_table],
            )

            # Examples
            gr.Examples(
                examples=[
                    [8, 25, 20, 1, 0.35, 0.28, 12, 9, 300, 250, 0.0005, 0.0004, 6, 5],
                    [2, 40, 5,  0, 0.05, 0.02, 1,  0, 800, 50,  0.0008, 0.0001, 8, 1],
                    [5, 15, 12, 1, 0.20, 0.22, 6,  5, 120, 100, 0.0002, 0.0003, 3, 3],
                    [0, 60, 3,  0, 0.01, 0.05, 0,  1, 900, 30,  0.0010, 0.0001, 9, 1],
                ],
                inputs=[cn, du, dv, same_comm, cu, cv, tu, tv, csu, csv_, pru, prv, kcu, kcv],
                label="📋 Quick Examples",
            )

        # ── Models Tab ────────────────────────────────────────────────────────
        with gr.TabItem("📊 Model Performance"):
            with gr.Row():
                auc_plot   = gr.Plot(label="AUC-ROC Comparison")
                radar_plot = gr.Plot(label="Multi-Metric Radar")
            model_table    = gr.DataFrame(label="Full Metrics", wrap=True)

            def _load_models():
                return build_model_charts()

            gr.Button("Load Model Results", variant="secondary").click(
                fn=_load_models, inputs=[], outputs=[auc_plot, radar_plot, model_table],
            )

        # ── Features Tab ─────────────────────────────────────────────────────
        with gr.TabItem("🧬 Feature Analysis"):
            with gr.Row():
                imp_plot = gr.Plot(label="Feature Importance")
                pie_plot = gr.Plot(label="Category Distribution")

            gr.Button("Load Feature Charts", variant="secondary").click(
                fn=build_feature_charts, inputs=[], outputs=[imp_plot, pie_plot],
            )

            gr.HTML("""
            <div class='nc-stitle' style='margin-top:24px;'>Feature Categories</div>
            <div style='display:flex;flex-wrap:wrap;gap:12px;margin-top:12px;'>
              <div style='flex:1;min-width:180px;background:rgba(94,184,255,.06);border:1px solid rgba(94,184,255,.2);border-radius:14px;padding:16px;'>
                <div style='color:#5eb8ff;font-weight:700;margin-bottom:8px;'>🌐 Neighborhood</div>
                <div style='color:#94a3c4;font-size:.8rem;font-family:JetBrains Mono,monospace;'>common_neighbors · jaccard · adamic_adar · resource_allocation · salton_index · sorensen_index</div>
              </div>
              <div style='flex:1;min-width:180px;background:rgba(183,148,244,.06);border:1px solid rgba(183,148,244,.2);border-radius:14px;padding:16px;'>
                <div style='color:#b794f4;font-weight:700;margin-bottom:8px;'>🏗️ Structural</div>
                <div style='color:#94a3c4;font-size:.8rem;font-family:JetBrains Mono,monospace;'>pref_attach · degree_u · degree_v · degree_diff · degree_ratio · degree_product_log</div>
              </div>
              <div style='flex:1;min-width:180px;background:rgba(45,212,191,.06);border:1px solid rgba(45,212,191,.2);border-radius:14px;padding:16px;'>
                <div style='color:#2dd4bf;font-weight:700;margin-bottom:8px;'>👥 Community</div>
                <div style='color:#94a3c4;font-size:.8rem;font-family:JetBrains Mono,monospace;'>same_community · comm_size_u · comm_size_ratio</div>
              </div>
              <div style='flex:1;min-width:180px;background:rgba(245,158,11,.06);border:1px solid rgba(245,158,11,.2);border-radius:14px;padding:16px;'>
                <div style='color:#f59e0b;font-weight:700;margin-bottom:8px;'>🔵 Clustering</div>
                <div style='color:#94a3c4;font-size:.8rem;font-family:JetBrains Mono,monospace;'>clustering_u · clustering_v · avg_clustering · triangles_u · triangles_v</div>
              </div>
              <div style='flex:1;min-width:180px;background:rgba(244,114,182,.06);border:1px solid rgba(244,114,182,.2);border-radius:14px;padding:16px;'>
                <div style='color:#f472b6;font-weight:700;margin-bottom:8px;'>📍 Centrality</div>
                <div style='color:#94a3c4;font-size:.8rem;font-family:JetBrains Mono,monospace;'>pagerank_u · pagerank_v · pagerank_ratio · core_u · core_v</div>
              </div>
            </div>
            """)

        # ── API Tab ───────────────────────────────────────────────────────────
        with gr.TabItem("⚙️ API & Deploy"):
            gr.Markdown("""
## 🚀 Deployment Options

### Hugging Face Spaces
Upload these files to your Space:
- `app.py` (this file)
- `pipeline.py`
- `requirements.txt`
- `models/best_model.pkl` *(optional — formula fallback if missing)*
- `summary.json` *(optional)*

Set **SDK = Gradio** in your Space settings.

---

### Render Cloud Deployment
```bash
# 1. Push project to GitHub
git push origin main

# 2. Create new Web Service on render.com
# Build Command:  pip install -r requirements.txt
# Start Command:  python gradio_app.py  (or gunicorn for Flask)
```

Or use the included `render.yaml` for one-click deploy.

---

### Streamlit Cloud
```bash
streamlit run streamlit_app.py
```
Upload to https://streamlit.io/cloud

---

### Flask REST API
```bash
cd flask_app
python app.py
# POST /predict  · POST /api/batch  · GET /api/stats
```

---

### Python Pipeline (direct use)
```python
from pipeline import predict

result = predict(
    cn=5, du=20, dv=15, same_community=1,
    clust_u=0.25, clust_v=0.18,
    tri_u=8, tri_v=6,
    cs_u=200, cs_v=150,
    pr_u=0.0003, pr_v=0.0002,
    core_u=4, core_v=3,
)
print(result.score, result.verdict)
```
""")

    # ── Footer ─────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class='nc-divider' style='margin-top:32px;'></div>
    <div style='text-align:center;color:#4a5570;font-size:.78rem;font-family:JetBrains Mono,monospace;padding-bottom:24px;'>
      NeuroCollab v3.0 MAX · Graph ML · DBLP · Voting Ensemble · ~99.4% AUC-ROC
    </div>
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
    )
