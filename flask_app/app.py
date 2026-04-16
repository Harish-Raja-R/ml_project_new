"""
NeuroCollab — Flask API v3.0 MAX
Powered by the shared pipeline.py module.

Endpoints:
  GET  /            → Main glassmorphism web UI
  POST /predict     → Single pair prediction
  POST /api/batch   → Batch prediction (≤50 pairs)
  GET  /api/stats   → Model metrics + feature importance
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, request, render_template, jsonify
from pipeline import (
    predict, batch_predict, get_model_results, get_feature_importance,
    get_best_model_name, get_summary, FEATURE_NAMES,
)

app = Flask(__name__)

@app.template_filter('format_int')
def format_int_filter(value):
    try:
        return f"{int(value):,}"
    except:
        return str(value)


@app.route('/')
def index():
    summary = get_summary()
    stats = {
        'best_model':         get_best_model_name(),
        'num_features':       len(FEATURE_NAMES),
        'model_results':      get_model_results(),
        'feature_importance': get_feature_importance(),
        'graph_nodes':        summary.get('num_nodes', 20000),
        'graph_edges':        summary.get('num_edges', 15384),
        'total_samples':      summary.get('samples', {}).get('total', 12000),
        'using_ml_model':     True,
        'version':            summary.get('version', '3.0-MAX'),
    }
    return render_template('index.html', stats=stats)


@app.route('/predict', methods=['POST'])
def predict_route():
    d = request.get_json()
    result = predict(
        cn=int(d.get('common_neighbors', 0)),
        du=int(d.get('degree_u', 5)),
        dv=int(d.get('degree_v', 5)),
        same_community=int(d.get('same_community', 0)),
        clust_u=float(d.get('clust_u', 0.1)),
        clust_v=float(d.get('clust_v', 0.1)),
        tri_u=int(d.get('tri_u', 2)),
        tri_v=int(d.get('tri_v', 2)),
        cs_u=int(d.get('cs_u', 100)),
        cs_v=int(d.get('cs_v', 100)),
        pr_u=float(d.get('pr_u', 0.0001)),
        pr_v=float(d.get('pr_v', 0.0001)),
        core_u=int(d.get('core_u', 2)),
        core_v=int(d.get('core_v', 2)),
    )
    return jsonify(result.to_dict())


@app.route('/api/stats')
def api_stats():
    summary = get_summary()
    return jsonify({
        'model_results':      get_model_results(),
        'best_model':         get_best_model_name(),
        'feature_importance': get_feature_importance(),
        'feature_names':      FEATURE_NAMES,
        'graph_stats': {
            'nodes': summary.get('num_nodes', 20000),
            'edges': summary.get('num_edges', 15384),
        },
        'version': summary.get('version', '3.0-MAX'),
    })


@app.route('/api/batch', methods=['POST'])
def api_batch():
    data    = request.get_json()
    pairs   = data.get('pairs', [])
    results = batch_predict(pairs)
    return jsonify({'results': results, 'count': len(results)})


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
