"""
Phishing Detection - Feature Extractor
Extracts URL-based and webpage-based features for classification.
"""

import re
import math
import urllib.parse
from urllib.parse import urlparse


# ── Suspicious token lists ──────────────────────────────────────────────────

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "account", "update", "secure", "banking",
    "paypal", "amazon", "apple", "microsoft", "google", "facebook", "netflix",
    "confirm", "password", "credential", "wallet", "crypto", "support",
    "helpdesk", "alert", "suspended", "unlock", "validate", "free", "prize",
    "winner", "click", "limited", "offer", "urgent", "invoice", "refund",
    "ebay", "wellsfargo", "chase", "citibank", "irs", "gov", "official",
]

BRAND_NAMES = [
    "paypal", "amazon", "apple", "microsoft", "google", "facebook",
    "netflix", "ebay", "chase", "wellsfargo", "citibank", "instagram",
    "twitter", "linkedin", "dropbox", "outlook", "office365", "onedrive",
    "icloud", "yahoo", "bing", "adobe", "docusign", "zoom",
]

TRUSTED_TLDS = {".com", ".org", ".net", ".edu", ".gov", ".mil", ".int"}
SUSPICIOUS_TLDS = {
    ".xyz", ".tk", ".ml", ".ga", ".cf", ".gq", ".top", ".click", ".link",
    ".work", ".date", ".racing", ".stream", ".download", ".win", ".bid",
    ".party", ".review", ".trade", ".accountant", ".loan", ".men",
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _entropy(s: str) -> float:
    """Shannon entropy of a string."""
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def _count_special(s: str) -> int:
    return sum(1 for c in s if not c.isalnum())


def _has_ip(hostname: str) -> bool:
    ipv4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
    return bool(ipv4.match(hostname))


# ── Main extractor ───────────────────────────────────────────────────────────

def extract_url_features(url: str) -> dict:
    """Return a flat dict of numeric features derived from the URL string."""

    # Normalise
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""
    full_url = url.lower()

    # ── Length features ──────────────────────────────────────────────────────
    f = {}
    f["url_length"] = len(url)
    f["hostname_length"] = len(hostname)
    f["path_length"] = len(path)
    f["query_length"] = len(query)

    # ── Character count features ─────────────────────────────────────────────
    f["dot_count"] = url.count(".")
    f["hyphen_count"] = url.count("-")
    f["underscore_count"] = url.count("_")
    f["slash_count"] = url.count("/")
    f["at_count"] = url.count("@")          # @ → credential spoofing
    f["question_count"] = url.count("?")
    f["equal_count"] = url.count("=")
    f["ampersand_count"] = url.count("&")
    f["percent_count"] = url.count("%")     # URL encoding tricks
    f["digit_count"] = sum(c.isdigit() for c in url)
    f["special_char_count"] = _count_special(url)

    # ── Structural features ──────────────────────────────────────────────────
    f["is_https"] = int(parsed.scheme == "https")
    f["has_port"] = int(parsed.port is not None and parsed.port not in (80, 443))
    f["has_ip_address"] = int(_has_ip(hostname))
    f["subdomain_count"] = max(0, len(hostname.split(".")) - 2)
    f["path_depth"] = len([p for p in path.split("/") if p])

    # TLD features
    parts = hostname.rsplit(".", 1)
    tld = "." + parts[-1] if len(parts) > 1 else ""
    f["suspicious_tld"] = int(tld in SUSPICIOUS_TLDS)
    f["trusted_tld"] = int(tld in TRUSTED_TLDS)

    # ── Entropy features ─────────────────────────────────────────────────────
    f["hostname_entropy"] = round(_entropy(hostname), 4)
    f["url_entropy"] = round(_entropy(url), 4)

    # ── Keyword / brand features ─────────────────────────────────────────────
    f["suspicious_keyword_count"] = sum(
        1 for kw in SUSPICIOUS_KEYWORDS if kw in full_url
    )
    brand_hits = [b for b in BRAND_NAMES if b in hostname]
    f["brand_in_hostname"] = int(len(brand_hits) > 0)
    f["brand_count_in_url"] = sum(1 for b in BRAND_NAMES if b in full_url)

    # ── Obfuscation signals ──────────────────────────────────────────────────
    f["has_at_symbol"] = int("@" in url)
    f["has_double_slash_redirect"] = int("//" in path)
    f["has_hex_encoding"] = int(bool(re.search(r"%[0-9a-fA-F]{2}", url)))
    f["has_punycode"] = int("xn--" in hostname)
    f["digit_ratio_in_hostname"] = round(
        sum(c.isdigit() for c in hostname) / max(len(hostname), 1), 4
    )

    # ── URL shortener ────────────────────────────────────────────────────────
    shorteners = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
                  "is.gd", "buff.ly", "adf.ly", "short.link", "rebrand.ly"}
    f["is_shortened"] = int(any(s in hostname for s in shorteners))

    # ── Query-string suspicion ───────────────────────────────────────────────
    f["query_param_count"] = len(urllib.parse.parse_qs(query))
    f["long_query"] = int(len(query) > 100)

    return f


def features_to_vector(feat_dict: dict) -> list:
    """Return an ordered list of feature values (same order every call)."""
    return [feat_dict[k] for k in sorted(feat_dict.keys())]


def feature_names() -> list:
    """Return sorted feature names (must match features_to_vector order)."""
    sample = extract_url_features("http://example.com")
    return sorted(sample.keys())

