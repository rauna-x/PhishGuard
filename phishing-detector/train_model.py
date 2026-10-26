"""
Phishing Detection - Model Trainer
Trains a Random Forest + Gradient Boosting ensemble on synthetic + heuristic data,
then serialises the pipeline to disk.
"""

import os
import random
import math
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from feature_extractor import extract_url_features, features_to_vector, feature_names

# ─── Seed for reproducibility ────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "phishing_model.pkl")

# ─── Synthetic dataset ────────────────────────────────────────────────────────

LEGITIMATE_URLS = [
    "https://www.google.com/search?q=weather",
    "https://www.wikipedia.org/wiki/Phishing",
    "https://github.com/features",
    "https://stackoverflow.com/questions",
    "https://www.amazon.com/dp/B08N5WRWNW",
    "https://www.bbc.com/news",
    "https://www.nytimes.com/",
    "https://www.reddit.com/r/programming",
    "https://www.linkedin.com/in/someuser",
    "https://www.apple.com/iphone",
    "https://www.microsoft.com/en-us/windows",
    "https://www.netflix.com/browse",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.facebook.com/",
    "https://twitter.com/openai",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://www.python.org/downloads/",
    "https://docs.python.org/3/library/",
    "https://www.cloudflare.com/",
    "https://www.shopify.com/",
    "https://www.salesforce.com/",
    "https://zoom.us/j/meeting",
    "https://mail.google.com/mail/u/0/",
    "https://drive.google.com/drive/my-drive",
    "https://calendar.google.com/",
    "https://www.coursera.org/courses",
    "https://www.udemy.com/",
    "https://www.stripe.com/",
    "https://www.twilio.com/",
    "https://www.digitalocean.com/",
    "https://www.heroku.com/",
    "https://www.mongodb.com/",
    "https://www.mysql.com/",
    "https://www.postgresql.org/",
    "https://pypi.org/project/requests/",
    "https://npmjs.com/package/react",
    "https://www.w3schools.com/html/",
    "https://developer.mozilla.org/en-US/",
    "https://www.espn.com/",
    "https://www.weather.com/",
    "https://www.ebay.com/sch/i.html?_nkw=laptop",
    "https://www.walmart.com/browse/electronics",
    "https://www.target.com/",
    "https://www.bestbuy.com/",
    "https://www.homedepot.com/",
    "https://www.indeed.com/jobs",
    "https://glassdoor.com/",
    "https://www.yelp.com/",
    "https://www.tripadvisor.com/",
    "https://booking.com/",
]

PHISHING_URLS = [
    "http://paypa1-secure-login.xyz/account/verify",
    "http://192.168.1.105/phishing/login.php",
    "http://amazon-security-alert.tk/update-account",
    "http://microsoft-support-helpdesk.ml/verify-identity",
    "http://apple-id-locked.ga/unlock-now",
    "http://login-facebook-confirm.cf/credential",
    "http://secure-banking-alert.gq/signin",
    "http://netflix-billing-update.xyz/payment",
    "http://ebay-suspended-account.tk/restore",
    "http://google-prize-winner.ml/claim-now",
    "http://paypal.com.evil-site.ru/login",
    "http://update-your-account-info.click/verify",
    "http://free-iphone-winner.work/collect",
    "http://bit.ly/3xFree-Gift-Card",
    "http://citibank-login-secure.top/account",
    "http://irs-refund-alert.stream/verify",
    "http://wellsfargo-support.bid/secure",
    "http://chase-account-verify.date/login",
    "http://instagram-security.racing/unlock",
    "http://linkedin-job-offer-click.download/apply",
    "http://dropbox-share.party/document",
    "http://office365-verify.review/credential",
    "http://zoom-meeting-alert.loan/join-now",
    "http://icloud-locked.men/restore",
    "http://yahoo-prize-offer.win/claim",
    "http://secure.paypa1.com-verify.net/login",
    "http://amazon.account-update.suspicious-domain.org/verify",
    "http://xn--pypal-4ve.com/",
    "https://apple-id.c0m.phishingdomain.tk/login",
    "http://microsoft.com-helpdesk-support.xyz/reset",
    "http://123.45.67.89/banking/login",
    "http://suspicious-site.com/login?redirect=http://evil.com&token=abc123&session=xyz789",
    "http://free-gift-cards-no-survey.click/win",
    "http://urgent-account-suspended-verify-now.ml/",
    "http://your-account-has-been-hacked-verify.ga/",
    "http://confirm-your-identity-paypal-secure.cf/",
    "http://reactivate-your-netflix-account.xyz/billing",
    "http://amazon-order-problem-click-here.gq/track",
    "http://apple.com-account-locked-id12345.tk/",
    "http://secure-login-bank.xyz/credential/verify?id=12345&token=abcdef",
    "http://faceb00k-security-alert.ml/login",
    "http://g00gle-verify-account.cf/",
    "http://paypall.com/signin",
    "http://bankofamerica-alert.top/login",
    "http://irs.gov-taxrefund.tk/claim",
    "http://support-microsoft-helpdesk.ml/fix",
    "http://ebay-won-prize.xyz/winner",
    "http://claim-your-reward-click-now.link/",
    "http://crypto-wallet-verify-now.win/connect",
    "http://phish.example.xyz/%70%61%73%73%77%6F%72%64",  # hex encoded "password"
]

# Augment with more synthetic patterns
def _generate_phishing():
    brands = ["paypal", "amazon", "apple", "microsoft", "netflix", "google",
              "facebook", "ebay", "instagram", "linkedin"]
    actions = ["login", "verify", "confirm", "secure", "update", "alert", "support"]
    tlds = [".xyz", ".tk", ".ml", ".ga", ".cf", ".gq", ".top", ".click", ".work"]
    extras = ["/account/verify", "/login", "/signin", "/update", "/credential"]
    brand = random.choice(brands)
    action = random.choice(actions)
    tld = random.choice(tlds)
    extra = random.choice(extras)
    num = random.randint(1, 9999)
    return f"http://{brand}-{action}-{num}{tld}{extra}"

def _generate_legit():
    domains = ["example.com", "shopify.com", "medium.com", "notion.so",
               "figma.com", "vercel.app", "netlify.app", "railway.app"]
    paths = ["/blog", "/products", "/about", "/contact", "/pricing", "/docs"]
    domain = random.choice(domains)
    path = random.choice(paths)
    return f"https://www.{domain}{path}"

for _ in range(300):
    PHISHING_URLS.append(_generate_phishing())
for _ in range(200):
    LEGITIMATE_URLS.append(_generate_legit())


# ─── Build dataset ────────────────────────────────────────────────────────────

def build_dataset():
    X, y = [], []
    for url in LEGITIMATE_URLS:
        try:
            feat = extract_url_features(url)
            X.append(features_to_vector(feat))
            y.append(0)
        except Exception:
            pass
    for url in PHISHING_URLS:
        try:
            feat = extract_url_features(url)
            X.append(features_to_vector(feat))
            y.append(1)
        except Exception:
            pass
    return np.array(X, dtype=float), np.array(y, dtype=int)


# ─── Train & save ─────────────────────────────────────────────────────────────

def train_and_save():
    print("Building dataset ...")
    X, y = build_dataset()
    print(f"  Samples: {len(X)}  |  Phishing: {y.sum()}  |  Legit: {(y==0).sum()}")

    rf = RandomForestClassifier(n_estimators=200, max_depth=12,
                                min_samples_leaf=2, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                    max_depth=5, random_state=42)
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        voting="soft",
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", ensemble),
    ])

    print("Running 5-fold CV ...")
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1")
    print(f"  CV F1: {scores.mean():.4f} ± {scores.std():.4f}")

    print("Fitting final model ...")
    pipeline.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"  Model saved -> {MODEL_PATH}")
    return pipeline


if __name__ == "__main__":
    train_and_save()