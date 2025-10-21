# NetSense – Simulated Real-Time Network Anomaly Detection (Streamlit)
# Run: streamlit run netsense_demo.py

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import streamlit as st

st.set_page_config(
    page_title="NetSense – Anomaly Detection (Simulation)",
    page_icon="🔎",
    layout="wide",
)

# ----------------------------
# Styling (clean + minimal)
# ----------------------------
CUSTOM_CSS = """
<style>
    .metric-small .stMetric { padding: 0.25rem 0.5rem; }
    .anomaly-row { background-color: rgba(255, 0, 0, 0.08); }
    .normal-row { background-color: rgba(0, 128, 0, 0.05); }
    .footer-note { color:#667; font-size:0.85rem; }
    .good {color:#1a7f37; font-weight:600;}
    .warn {color:#b34700; font-weight:600;}
    .bad  {color:#b00020; font-weight:600;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------
# Sidebar: About
# ----------------------------
with st.sidebar:
    st.header("NetSense (Demo)")
    st.write(
        "Simulated network stream processed by an **IsolationForest** anomaly detector. "
        "Shows how our ML pipeline flags unusual traffic patterns and surfaces them to a web dashboard."
    )
    st.write("This is a **simulation** (no Raspberry Pi required).")
    st.caption("Model: scikit-learn IsolationForest | Data: synthetic packets")

# ----------------------------
# Helpers
# ----------------------------
SRC_POOL = np.array([f"10.0.0.{i}" for i in range(2, 30)])
DST_POOL = np.array(["8.8.8.8", "1.1.1.1", "172.16.0.2", "10.0.0.1", "192.168.0.5"])
PROTO_POOL = np.array(["TCP", "UDP", "ICMP"])

def generate_batch(n=120, spike=False):
    """Generate a batch of synthetic packets. If spike=True, create bursty/anomalous sizes."""
    src = np.random.choice(SRC_POOL, n)
    dst = np.random.choice(DST_POOL, n)
    proto = np.random.choice(PROTO_POOL, n, p=[0.7, 0.25, 0.05])  # mostly TCP
    base_sizes = np.random.randint(60, 1000, n)

    if spike:
        # Inject a burst of larger packet sizes + odd ports to simulate anomaly
        burst_idx = np.random.choice(np.arange(n), size=max(8, n // 10), replace=False)
        base_sizes[burst_idx] = np.random.randint(1100, 1500, len(burst_idx))

    sport = np.random.randint(1024, 65535, n)
    dport = np.random.randint(20, 8080, n)

    ts = pd.Timestamp.utcnow().round("S")

    df = pd.DataFrame({
        "timestamp": ts,
        "src_ip": src,
        "dst_ip": dst,
        "protocol": proto,
        "src_port": sport,
        "dst_port": dport,
        "packet_size": base_sizes
    })
    return df

def fit_model(df_hist):
    """Fit IsolationForest on numeric features from historical window."""
    feats = df_hist[["packet_size", "src_port", "dst_port"]].copy()
    # Simple scale by ranks to reduce variance sensitivity (quick & dirty)
    feats = feats.rank(pct=True)
    model = IsolationForest(n_estimators=150, contamination=0.08, random_state=42)
    model.fit(feats)
    return model

def predict(model, df_new):
    feats = df_new[["packet_size", "src_port", "dst_port"]].copy().rank(pct=True)
    pred = model.predict(feats)  # 1 normal, -1 anomaly
    score = model.decision_function(feats)
    out = df_new.copy()
    out["prediction"] = np.where(pred == -1, "Anomaly", "Normal")
    out["score"] = score
    return out

def summarize(df):
    total = len(df)
    anomalies = int((df["prediction"] == "Anomaly").sum())
    normal = total - anomalies
    pct = (anomalies / total * 100) if total else 0
    return total, anomalies, normal, round(pct, 1)

def simple_explanation(row: pd.Series) -> str:
    reasons = []
    if row["packet_size"] > 1200: reasons.append("unusually large packet")
    if row["dst_port"] in (22, 23, 3389): reasons.append("sensitive destination port")
    if row["protocol"] == "ICMP": reasons.append("ICMP burst")
    if not reasons:
        reasons.append("pattern deviates from recent baseline")
    return f"{'; '.join(reasons)}."

# ----------------------------
# Session state
# ----------------------------
if "history" not in st.session_state:
    # Seed with a decent training window
    hist = pd.concat([
        generate_batch(400, spike=False),
        generate_batch(120, spike=True),
    ], ignore_index=True)
    st.session_state.history = hist

if "model" not in st.session_state:
    st.session_state.model = fit_model(st.session_state.history.tail(800))

# Auto-refresh every 2 seconds to simulate stream
st_autorefresh = st.experimental_rerun  # fallback alias
st_autorefresh_placeholder = st.empty()
# Streamlit 1.37+: st.autorefresh exists; if not, use timer below
try:
    st_autorefresh = st.autorefresh  # type: ignore
except Exception:
    pass

if callable(st_autorefresh):
    st_autorefresh(interval=2000, key="netsense_refresh")

# ----------------------------
# Generate a new incoming batch with occasional spike
# ----------------------------
with st.spinner("Processing new packets..."):
    spike_now = np.random.rand() < 0.25
    new_batch = generate_batch(120, spike=spike_now)
    preds = predict(st.session_state.model, new_batch)
    # Append to history
    st.session_state.history = pd.concat([st.session_state.history, preds], ignore_index=True)
    # Keep history bounded
    st.session_state.history = st.session_state.history.tail(4000)

# Re-fit model every few refreshes to adapt baseline (lightweight)
if np.random.rand() < 0.2:
    st.session_state.model = fit_model(st.session_state.history.tail(1200))

# ----------------------------
# Top: Title + key metrics
# ----------------------------
st.title("NetSense – Network Anomaly Detection (Simulation)")

colA, colB, colC, colD = st.columns([1,1,1,1])
total, anoms, normal, pct = summarize(st.session_state.history.tail(300))
with colA: st.metric("Packets (last window)", total)
with colB: st.metric("Anomalies (last window)", anoms)
with colC: st.metric("Normal (last window)", normal)
with colD: st.metric("Anomaly rate", f"{pct}%")

# ----------------------------
# Charts
# ----------------------------
left, right = st.columns([1.25, 1])

# Trend of anomalies over recent windows
with left:
    st.subheader("Traffic & Anomalies Over Time")
    recent = st.session_state.history.tail(1200).copy()
    # aggregate per timestamp
    agg = recent.groupby(["timestamp", "prediction"]).size().unstack(fill_value=0)
    agg = agg.rename(columns={"Anomaly": "anomaly", "Normal": "normal"})
    st.line_chart(agg)

# Protocol distribution (recent)
with right:
    st.subheader("Protocol Mix (recent)")
    proto_counts = st.session_state.history.tail(1200)["protocol"].value_counts().to_frame("count")
    st.bar_chart(proto_counts)

# ----------------------------
# Live table (latest packets)
# ----------------------------
st.subheader("Latest Packets")
latest = st.session_state.history.tail(20).copy()
# Order for readability
latest = latest[[
    "timestamp", "src_ip", "dst_ip", "protocol", "src_port", "dst_port", "packet_size", "prediction", "score"
]].sort_values("timestamp", ascending=False)

# Add short explanation for anomalies
latest["explain"] = latest.apply(lambda r: simple_explanation(r) if r["prediction"] == "Anomaly" else "", axis=1)

# Style rows for quick scanning
def row_style(row):
    return ['anomaly-row' if row["prediction"] == "Anomaly" else 'normal-row'] * len(row)

st.dataframe(latest.style.apply(row_style, axis=1), use_container_width=True, height=360)

# ----------------------------
# Footer note
# ----------------------------
st.markdown(
    "<div class='footer-note'>"
    "This demo simulates a live feed. In production, packets are captured on a Raspberry Pi, "
    "stored in a database, scored by an ML model, and surfaced to a web UI just like this."
    "</div>",
    unsafe_allow_html=True
)

