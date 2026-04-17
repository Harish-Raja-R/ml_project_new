
👉 This mismatch will break rendering.

---

## ✅ FINAL CORRECT VERSION (Fixed)

Just copy this 👇 (only formatting fixed, content untouched)

---

```markdown
# 🚀 NeuroCollab — Academic Collaboration Predictor v3.0 MAX

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![License Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?style=flat-square)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)
[![Accuracy 97.1%](https://img.shields.io/badge/Accuracy-97.1%25-brightgreen.svg?style=flat-square)]()
[![AUC-ROC 99.8%](https://img.shields.io/badge/AUC--ROC-99.8%25-blue.svg?style=flat-square)]()
[![Stars](https://img.shields.io/github/stars/Harish-Raja-R/neurocollab.svg?style=social&label=Stars)](https://github.com/Harish-Raja-R/neurocollab)
[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?style=flat-square)](NeuroCollab_Tutorial.ipynb)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-blue.svg?style=flat-square)]()

**Graph Machine Learning · 25 Features · Voting Ensemble · Flask + Streamlit + Gradio · HF Spaces Ready**

### 🎯 [→ Try Live Demo](https://huggingface.co/spaces/harish-raja/neurocollab) | [📖 Tutorial Notebook](NeuroCollab_Tutorial.ipynb) | [🌍 GitHub](https://github.com/Harish-Raja-R/neurocollab)

</div>

---

## 📋 About

**NeuroCollab** predicts academic collaboration compatibility using **graph-based machine learning** on the DBLP co-authorship network.

It analyzes:
- **317K+ researchers**
- **1M+ collaboration edges**

Achieves:
- **97.1% accuracy**

---

<details open>
<summary><b>💡 Why NeuroCollab?</b></summary>

| Aspect | Details |
|--------|--------|
| 🎯 Problem | Finding good research collaborators in large networks is manual and inefficient |
| ✨ Solution | ML predicts collaboration compatibility from graph topology + network analysis |
| 📊 Dataset | Real Stanford SNAP DBLP network |
| 🚀 Performance | 97.1% accuracy |
| ☁️ Deployable | Local Flask + Cloud HF Spaces |
| 🔬 Production | Scikit-Learn Pipelines + Voting Ensemble |

</details>

---

## 🎯 Key Statistics

```text
📈 DATASET SCALE        │ 🤖 MODEL PERFORMANCE      │ 🎨 FEATURE ENGINEERING
─────────────────────────────────────────────────────────────────────────────
Researchers:   317,080  │ Accuracy:   97.1%        │ Total Features:   25
Edges:       1,049,866  │ F1-Score:   97.1%        │ Neighborhood:      6
Communities:    13,477  │ AUC-ROC:    99.8%        │ Structural:        6
Avg Degree:        6.62 │ MCC:        94.3%        │ Community:         3
Network Dia:         22 │ Balanced:   97.1%        │ Clustering:        5

---

## 🔥 Use Cases

| Use Case              | Example            | Benefit              |
| --------------------- | ------------------ | -------------------- |
| 🔍 Partner Discovery  | Find researchers   | Faster collaboration |
| 🎓 Advisor Matching   | Match PhD students | Better fit           |
| 📈 Network Analysis   | Study topology     | Insights             |
| 🌍 Social Networks    | Extend model       | Predict connections  |
| 🤝 Community Building | Identify gaps      | Strong networks      |

---

## ⚡ Quick Start

### 🌐 Try Online

👉 [https://huggingface.co/spaces/harish-raja/neurocollab](https://huggingface.co/spaces/harish-raja/neurocollab)

---

### 💻 Run Locally

```bash
git clone https://github.com/Harish-Raja-R/neurocollab.git
cd neurocollab
pip install -r requirements.txt
```

```bash
# Flask
cd flask_app && python app.py

# Streamlit
streamlit run streamlit_app.py

# Gradio
python gradio_app.py
```

---

## 🐍 Python Usage

```python
from pipeline import predict

result = predict(
    cn=5, du=20, dv=15, same_community=1
)

print(result.score)
print(result.verdict)
```

---

## ✨ Features

* Voting Ensemble Model
* Explainable AI
* Flask + Streamlit + Gradio
* 25 Graph Features

---

## 📊 Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 85.2%    |
| Random Forest       | 96.8%    |
| XGBoost             | 97.2%    |
| LightGBM            | 97.3%    |
| Voting Ensemble     | 97.1%    |

---

## 📦 Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 API Example

```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{
  "common_neighbors": 5,
  "degree_u": 20,
  "degree_v": 15,
  "same_community": 1
}'
```

---

## 📄 License

Apache 2.0

```
