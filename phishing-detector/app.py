"""
Phishing Detection - Flask Web Application
Serves the UI and exposes /api/analyze for URL classification.
"""

import os
import json
import time
import threading
import joblib
import re
import urllib.parse
from urllib.parse import urlparse

from flask import Flask, request, jsonify, render_template, send_from_directory

from feature_extractor import extract_url_features, features_to_vector, feature_names

# ─── App Setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JSON_SORT_KEYS"] = False

MODEL_PATH = os.path.join(BASE_DIR, "models", "phishing_model.pkl")
_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            if os.path.exists(MODEL_PATH):
                _model = joblib.load(MODEL_PATH)
            else:
                return None
    return _model


# ─── Risk helpers ─────────────────────────────────────────────────────────────

RISK_EXPLANATIONS = {
    "has_ip_address":          ("Uses IP address instead of domain name",      "high"),
    "suspicious_tld":          ("Uses suspicious top-level domain",             "high"),
    "has_at_symbol":           ("Contains @ symbol (credential spoofing risk)", "high"),
    "has_punycode":            ("Uses Punycode encoding (homograph attack)",    "high"),
    "has_hex_encoding":        ("Contains hex-encoded characters",              "medium"),
    "is_shortened":            ("Uses a URL shortener (hides true destination)","medium"),
    "has_double_slash_redirect":("Contains double-slash redirect",              "medium"),
    "brand_in_hostname":       ("Brand name appears in hostname (spoofing)",    "high"),
    "suspicious_keyword_count":("Contains suspicious keywords",                 "medium"),
    "subdomain_count":         ("Excessive subdomains",                         "medium"),
    "has_port":                ("Uses non-standard port",                       "medium"),
    "long_query":              ("Unusually long query string",                  "low"),
    "dot_count":               ("Many dots in URL",                             "low"),
    "hyphen_count":            ("Many hyphens in URL",                          "low"),
    "url_length":              ("Unusually long URL",                           "low"),
}

POSITIVE_SIGNALS = {
    "is_https":    "Uses HTTPS encryption",
    "trusted_tld": "Uses a trusted top-level domain",
}


def build_risk_flags(features: dict) -> list:
    flags = []
    f = features

    if f.get("has_ip_address"):
        flags.append({"label": RISK_EXPLANATIONS["has_ip_address"][0], "severity": "high"})
    if f.get("suspicious_tld"):
        flags.append({"label": RISK_EXPLANATIONS["suspicious_tld"][0], "severity": "high"})
    if f.get("has_at_symbol"):
        flags.append({"label": RISK_EXPLANATIONS["has_at_symbol"][0], "severity": "high"})
    if f.get("has_punycode"):
        flags.append({"label": RISK_EXPLANATIONS["has_punycode"][0], "severity": "high"})
    if f.get("brand_in_hostname") and f.get("suspicious_tld"):
        flags.append({"label": "Brand name + suspicious TLD = likely spoofing", "severity": "high"})
    if f.get("has_hex_encoding"):
        flags.append({"label": RISK_EXPLANATIONS["has_hex_encoding"][0], "severity": "medium"})
    if f.get("is_shortened"):
        flags.append({"label": RISK_EXPLANATIONS["is_shortened"][0], "severity": "medium"})
    if f.get("has_double_slash_redirect"):
        flags.append({"label": RISK_EXPLANATIONS["has_double_slash_redirect"][0], "severity": "medium"})
    if f.get("suspicious_keyword_count", 0) >= 2:
        flags.append({
            "label": f"Contains {f['suspicious_keyword_count']} suspicious keywords",
            "severity": "medium"
        })
    if f.get("subdomain_count", 0) >= 3:
        flags.append({
            "label": f"Excessive subdomains ({f['subdomain_count']})",
            "severity": "medium"
        })
    if f.get("has_port"):
        flags.append({"label": RISK_EXPLANATIONS["has_port"][0], "severity": "medium"})
    if f.get("url_length", 0) > 100:
        flags.append({"label": f"Very long URL ({f['url_length']} chars)", "severity": "low"})
    if f.get("hyphen_count", 0) >= 4:
        flags.append({
            "label": f"Many hyphens in URL ({f['hyphen_count']})",
            "severity": "low"
        })
    if f.get("long_query"):
        flags.append({"label": RISK_EXPLANATIONS["long_query"][0], "severity": "low"})
    if not f.get("is_https"):
        flags.append({"label": "Does not use HTTPS", "severity": "medium"})

    return flags


def build_positive_signals(features: dict) -> list:
    sigs = []
    if features.get("is_https"):
        sigs.append("Uses HTTPS (encrypted connection)")
    if features.get("trusted_tld"):
        sigs.append("Trusted top-level domain (.com / .org / .net / .edu / .gov)")
    if not features.get("has_ip_address"):
        sigs.append("Uses a human-readable domain name")
    if features.get("subdomain_count", 0) <= 1:
        sigs.append("Simple subdomain structure")
    if features.get("suspicious_keyword_count", 0) == 0:
        sigs.append("No suspicious keywords detected")
    return sigs


def score_to_label(prob: float) -> tuple:
    """(verdict, color_class, description)"""
    if prob >= 0.80:
        return "Phishing", "danger", "High probability of phishing — do NOT visit."
    if prob >= 0.55:
        return "Suspicious", "warning", "Suspicious signals detected — proceed with extreme caution."
    if prob >= 0.35:
        return "Uncertain", "uncertain", "Some risk signals present — verify before sharing data."
    return "Safe", "safe", "No major phishing indicators detected."


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()

    if not url:
        return jsonify({"error": "No URL provided"}), 400
    if len(url) > 2048:
        return jsonify({"error": "URL too long (max 2048 chars)"}), 400

    # Normalise
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    try:
        features = extract_url_features(url)
    except Exception as e:
        return jsonify({"error": f"Feature extraction failed: {e}"}), 500

    model = get_model()
    if model is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 503

    try:
        vec = features_to_vector(features)
        prob_phish = float(model.predict_proba([vec])[0][1])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    verdict, color, description = score_to_label(prob_phish)
    risk_flags = build_risk_flags(features)
    positive = build_positive_signals(features)

    # Parse display info
    parsed = urlparse(url)
    hostname = parsed.hostname or url

    response = {
        "url": url,
        "hostname": hostname,
        "verdict": verdict,
        "color": color,
        "description": description,
        "phishing_probability": round(prob_phish * 100, 1),
        "risk_flags": risk_flags,
        "positive_signals": positive,
        "features": {
            "URL Length":        features["url_length"],
            "HTTPS":             bool(features["is_https"]),
            "IP Address Used":   bool(features["has_ip_address"]),
            "Subdomain Depth":   features["subdomain_count"],
            "Suspicious TLD":    bool(features["suspicious_tld"]),
            "Suspicious Keywords": features["suspicious_keyword_count"],
            "URL Entropy":       features["url_entropy"],
            "Hyphens":           features["hyphen_count"],
            "Dots":              features["dot_count"],
            "Hex Encoding":      bool(features["has_hex_encoding"]),
        }
    }
    return jsonify(response)


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": get_model() is not None,
        "feature_count": len(feature_names()),
    })


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  PhishGuard — Phishing Detection Server")
    print("=" * 55)
    model = get_model()
    if model:
        print("  ✓ Model loaded from disk")
    else:
        print("  ⚠ Model not found — run train_model.py first")
    print("  → http://localhost:5000")
    print("=" * 55)
    app.run(debug=False, port=5000)

