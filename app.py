import streamlit as st
import numpy as np

# --- Copying the logic from your notebook ---
risk_universe = np.linspace(0, 1, 100)
def mf_low(x): return np.maximum(0, np.minimum(1, (0.4 - x) / 0.4))
def mf_med(x): return np.maximum(0, np.minimum((x - 0.2) / 0.3, (0.8 - x) / 0.3))
def mf_high(x): return np.maximum(0, np.minimum(1, (x - 0.6) / 0.4))

def mamdani_raw(row):
    f = [np.clip(val, 0, 1) for val in row]
    rules = [
        (f[0], 2), (f[1], 2), (f[2], 1),
        (min(f[0], f[1]), 2), (min(f[3], f[4]), 1),
        (max(f[0], f[2]), 2), (min(f[1], f[4]), 2),
        (1-f[0], 0), (1-f[3], 0), (1-f[4], 0),
        (min(f[3], 0.5), 1), (min(f[0], 0.3), 0),
        (f[0]*f[1], 2), (f[3]*f[4], 1), (f[2]*f[4], 2)
    ]
    out_low, out_med, out_high = 0, 0, 0
    for strength, level in rules:
        if level == 0: out_low = max(out_low, strength)
        elif level == 1: out_med = max(out_med, strength)
        else: out_high = max(out_high, strength)
    num = np.sum(risk_universe * np.maximum.reduce([
        np.minimum(out_low, mf_low(risk_universe)),
        np.minimum(out_med, mf_med(risk_universe)),
        np.minimum(out_high, mf_high(risk_universe))
    ]))
    den = np.sum(np.maximum.reduce([
        np.minimum(out_low, mf_low(risk_universe)),
        np.minimum(out_med, mf_med(risk_universe)),
        np.minimum(out_high, mf_high(risk_universe))
    ])) + 1e-9
    return num / den

def sugeno_raw(row):
    f = [np.clip(val, 0, 1) for val in row]
    rules = [
        (f[0], 0.9), (f[1], 0.8), (f[2], 0.6),
        (min(f[0], f[1]), 0.95), (min(f[3], f[4]), 0.5),
        (max(f[0], f[2]), 0.85), (min(f[1], f[4]), 0.7),
        (1-f[0], 0.1), (1-f[3], 0.2), (1-f[4], 0.25),
        (min(f[3], 0.5), 0.4), (min(f[0], 0.3), 0.1),
        (f[0]*f[1], 0.9), (f[3]*f[4], 0.5), (f[2]*f[4], 0.8)
    ]
    num = sum(strength * output for strength, output in rules)
    den = sum(strength for strength, output in rules) + 1e-9
    return num / den

# --- Streamlit UI ---
st.title("Heart Disease Prediction - Fuzzy Logic")
st.write("Input patient data to calculate risk score.")

# Assuming inputs are scaled 0-1 as in your X_raw_eval
f0 = st.slider("Exercise Induced Angina (0-1)", 0.0, 1.0, 0.0)
f1 = st.slider("Smoking (0-1)", 0.0, 1.0, 0.0)
f2 = st.slider("Diabetes (0-1)", 0.0, 1.0, 0.0)
f3 = st.slider("Cholesterol (Scaled 0-1)", 0.0, 1.0, 0.5)
f4 = st.slider("Age (Scaled 0-1)", 0.0, 1.0, 0.5)

input_data = [f0, f1, f2, f3, f4]

if st.button("Predict"):
    m_res = mamdani_raw(input_data)
    s_res = sugeno_raw(input_data)
    
    col1, col2 = st.columns(2)
    col1.metric("Mamdani Risk", f"{m_res:.2f}")
    col2.metric("Sugeno Risk", f"{s_res:.2f}")
    
    status = "High Risk" if m_res > 0.5 else "Low Risk"
    st.subheader(f"Prediction Status: {status}")
