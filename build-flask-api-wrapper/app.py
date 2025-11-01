# app.py
import os
import traceback
from functools import wraps
from flask import Flask, request, jsonify
import joblib
import numpy as np
from werkzeug.exceptions import BadRequest

app = Flask(__name__)
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")

def load_model(path):
    payload = joblib.load(path)
    model = payload["model"]
    meta = {
        "feature_names": payload.get("feature_names"),
        "target_names": payload.get("target_names"),
        "model_version": payload.get("model_version", "unknown")
    }
    return model, meta

try:
    model, MODEL_META = load_model(MODEL_PATH)
    print("Model loaded:", MODEL_META)
except Exception as e:
    print("Failed to load model:", e)
    model, MODEL_META = None, {}

def json_endpoint(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except BadRequest as br:
            return jsonify({"error": "bad_request", "message": str(br.description)}), 400
        except Exception as e:
            tb = traceback.format_exc()
            app.logger.error(tb)
            return jsonify({"error": "internal_error", "message": str(e)}), 500
    return wrapped

@app.route("/health", methods=["GET"])
def health():
    ok = model is not None
    return jsonify({"status": "ok" if ok else "unavailable", "model_loaded": ok}), 200 if ok else 503

@app.route("/metadata", methods=["GET"])
def metadata():
    return jsonify({"model_meta": MODEL_META})

def validate_features(features):
    if not isinstance(features, list):
        raise BadRequest("features must be a list of numbers")
    if any(not isinstance(x, (int, float)) for x in features):
        raise BadRequest("each feature must be numeric")
    return np.array(features, dtype=float).reshape(1, -1)

@app.route("/predict", methods=["POST"])
@json_endpoint
def predict():
    payload = request.get_json(force=True)
    if payload is None:
        raise BadRequest("expected JSON body")
    if "features" in payload:
        features = payload["features"]
    else:
        if isinstance(payload, list):
            features = payload
        else:
            raise BadRequest("payload must include 'features' key or be a top-level list")
    X = validate_features(features)
    preds = model.predict(X)
    probs = None
    try:
        probs = model.predict_proba(X).tolist()
    except Exception:
        probs = None
    target_name = MODEL_META.get("target_names")
    pred_label = target_name[preds[0]] if target_name else int(preds[0])
    return jsonify({
        "prediction": int(preds[0]),
        "label": pred_label,
        "probabilities": probs,
        "model_version": MODEL_META.get("model_version")
    })

@app.route("/predict_batch", methods=["POST"])
@json_endpoint
def predict_batch():
    payload = request.get_json(force=True)
    if payload is None:
        raise BadRequest("expected JSON body")
    instances = payload.get("instances")
    if instances is None:
        if isinstance(payload, list):
            instances = payload
        else:
            raise BadRequest("payload must include 'instances' key or be a top-level list of feature lists")
    if not isinstance(instances, list) or not all(isinstance(r, list) for r in instances):
        raise BadRequest("instances must be a list of feature lists")
    X = np.array(instances, dtype=float)
    preds = model.predict(X).tolist()
    probs = None
    try:
        probs = model.predict_proba(X).tolist()
    except Exception:
        probs = None
    target_name = MODEL_META.get("target_names")
    labels = [target_name[int(p)] if target_name else int(p) for p in preds]
    return jsonify({
        "predictions": preds,
        "labels": labels,
        "probabilities": probs,
        "model_version": MODEL_META.get("model_version")
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
