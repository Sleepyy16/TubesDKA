import streamlit as st
import numpy as np

# --- Fuzzy Logic Functions ---
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
st.write("Silakan masukkan data pasien di bawah ini.")

# Inputs
col_a, col_b = st.columns(2)
with col_a:
    age = st.number_input("Umur (Tahun)", min_value=1, max_value=120, value=45)
    chol = st.number_input("Kolesterol (mg/dl)", min_value=100, max_value=600, value=200)
with col_b:
    smoking = st.selectbox("Apakah Pasien Merokok?", ["No", "Yes"])
    diabetes = st.selectbox("Apakah Pasien Diabetes?", ["No", "Yes"])
    exang = st.selectbox("Nyeri Dada Saat Olahraga? (Exercise Induced Angina)", ["No", "Yes"])

# Normalize inputs to 0-1 range for the model
# Assuming typical min-max from common heart datasets
f_age = (age - 20) / 60  # Scale age 20-80 to 0-1
f_chol = (chol - 120) / 300 # Scale chol 120-420 to 0-1
f_smoking = 1.0 if smoking == "Yes" else 0.0
f_diabetes = 1.0 if diabetes == "Yes" else 0.0
f_exang = 1.0 if exang == "Yes" else 0.0

# Internal feature order: [Exercise, Smoking, Diabetes, Cholesterol, Age]
input_data = [f_exang, f_smoking, f_diabetes, f_chol, f_age]

if st.button("Prediksi Risiko"):
    m_res = mamdani_raw(input_data)
    s_res = sugeno_raw(input_data)

    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Skor Mamdani", f"{m_res:.2%}")
    c2.metric("Skor Sugeno", f"{s_res:.2%}")

    if m_res > 0.6:
        st.error("Status: RISIKO TINGGI")
    elif m_res > 0.3:
        st.warning("Status: RISIKO SEDANG")
    else:
        st.success("Status: RISIKO RENDAH")
