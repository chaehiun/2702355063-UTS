# app.py
import streamlit as st
import pickle
import pandas as pd

# Load model
@st.cache_resource
def load_model():
    with open("best_xgb_model_new.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Title
st.title("Loan Approval Prediction")

# Input form
st.subheader("Masukkan Data Calon Peminjam:")
person_age = st.number_input("Umur", min_value=18, max_value=100, value=30)
person_gender = st.selectbox("Jenis Kelamin", ["male", "female"])
person_education = st.selectbox("Pendidikan", ["High School", "Bachelor", "Master","Associate", "Doctorate"])
person_income = st.number_input("Pendapatan per Tahun", value=50000)
person_emp_exp = st.slider("Pengalaman Kerja (tahun)", 0, 40, 5)
home_ownership = st.selectbox("Status Tempat Tinggal", ["OWN", "RENT", "OTHER"])
loan_amnt = st.number_input("Jumlah Pinjaman", value=10000)
loan_intent = st.selectbox("Tujuan Pinjaman", ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"])
loan_int_rate = st.number_input("Suku Bunga Pinjaman (%)", value=10.5)
loan_percent_income = loan_amnt / (person_income + 1e-6)
cb_person_cred_hist_length = st.number_input("Lama Riwayat Kredit (tahun)", value=3)
credit_score = st.number_input("Skor Kredit (credit_score)", min_value=300, max_value=850, value=600)
previous_loan_defaults_on_file = st.selectbox("Riwayat Gagal Bayar", ["No", "Yes"])

# Map categorical to numerical
gender_map = {"male": 1, "female": 0}
edu_map = {"High School": 0, "Bachelor": 1, "Master": 2,"Associate": 3, "Doctorate": 4}
default_map = {"No": 0, "Yes": 1}

loan_intents = ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]
home_ownerships = ["OTHER", "OWN", "RENT"]

input_data = {
    'person_age': person_age,
    'person_gender': gender_map[person_gender],
    'person_education': edu_map[person_education],
    'person_income': person_income,
    'person_emp_exp': person_emp_exp,
    'loan_amnt': loan_amnt,
    'loan_int_rate': loan_int_rate,
    'loan_percent_income': loan_percent_income,
    'cb_person_cred_hist_length': cb_person_cred_hist_length,
    'credit_score': credit_score,
    'previous_loan_defaults_on_file': default_map[previous_loan_defaults_on_file],
}

# One-hot encode loan_intent
for intent in loan_intents:
    input_data[f"loan_intent_{intent}"] = 1 if loan_intent == intent else 0

# One-hot encode home_ownership
for ho in home_ownerships:
    input_data[f"person_home_ownership_{ho}"] = 1 if home_ownership == ho else 0

# Predict
if st.button("Prediksi"):
    input_df = pd.DataFrame([input_data])
    result = model.predict(input_df)[0]
    st.success(f"Hasil Prediksi: {'DISETUJUI' if result == 1 else 'DITOLAK'}")

# Tambahkan test case
st.sidebar.header("💡 Test Case")
if st.sidebar.button("Test Case 1"):
    st.write("🔹 Laki-laki, Sarjana, Pendapatan: 60K, Pinjaman: 10K, Skor Kredit: Baik, Tujuan: PERSONAL")
if st.sidebar.button("Test Case 2"):
    st.write("🔹 Perempuan, SMA, Pendapatan: 20K, Pinjaman: 15K, Skor Kredit: Buruk, Tujuan: MEDICAL")
