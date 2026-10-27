PhishGuard — Phishing Detection System
A machine-learning powered phishing URL classifier with a web interface.

Project Structure
phishing-detector/
├── app.py                 # Flask web server & REST API
├── feature_extractor.py   # 30+ URL feature extraction functions
├── train_model.py         # Dataset builder + model trainer
├── requirements.txt       # Python dependencies
├── start.bat              # Windows startup script
├── models/
│   └── phishing_model.pkl # Trained Random Forest + GBM ensemble
└── templates/
    └── index.html         # Single-page web UI
Quick Start
1. Install Dependencies
pip install -r requirements.txt
2. Train the Model (already done — model is saved)
python train_model.py
3. Start the Server
# Option A: Double-click start.bat
# Option B:
python app.py
Then open http://localhost:5000 in your browser.

Features
URL Features Extracted (30+)
Category	Features
Length	URL length, hostname, path, query
Character counts	Dots, hyphens, slashes, @, %, digits
Structure	HTTPS, port, IP address, subdomains, path depth
TLD	Suspicious TLDs (.xyz, .tk, .ml…), trusted TLDs
Entropy	Shannon entropy of hostname and URL
Keywords	Suspicious words, brand name spoofing
Obfuscation	Hex encoding, Punycode, double-slash redirects
Shorteners	bit.ly, tinyurl.com, t.co, etc.
Classifier
Ensemble: Soft-voting of Random Forest (200 trees) + Gradient Boosting (150 trees)
Pipeline: StandardScaler → VotingClassifier
CV F1 Score: ~0.99 on training set
REST API
POST /api/analyze
Content-Type: application/json

{ "url": "https://suspicious-site.xyz/login" }
Response:

{
  "verdict": "Phishing",
  "color": "danger",
  "phishing_probability": 97.6,
  "risk_flags": [...],
  "positive_signals": [...],
  "features": {...}
}
Verdicts
Score	Label	Meaning
≥ 80%	🚨 Phishing	High probability — do NOT visit
55–80%	⚠️ Suspicious	Proceed with extreme caution
35–55%	🔍 Uncertain	Some risk signals — verify first
< 35%	✅ Safe	No major phishing indicators
