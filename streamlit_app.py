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
st.title("🎯 Prediksi Persetujuan Pinjaman (Loan Approval Prediction)")

# Input form
st.subheader("Masukkan Data Calon Peminjam:")

person_age = st.number_input("Usia (person_age)", min_value=18, max_value=100, value=30)
person_gender = st.selectbox("Jenis Kelamin (person_gender)", ["male", "female"])
person_education = st.selectbox("Pendidikan Terakhir (person_education)", ["High School", "Bachelor", "Master", "Associate", "Doctorate"])
person_income = st.number_input("Pendapatan Tahunan (USD) (person_income)", value=50000)
person_emp_exp = st.slider("Pengalaman Kerja (Tahun) (person_emp_exp)", 0, 40, 5)
person_home_ownership = st.selectbox("Kepemilikan Tempat Tinggal (person_home_ownership)", ["RENT", "OWN", "MORTGAGE", "OTHER"])
loan_amnt = st.number_input("Jumlah Pinjaman (loan_amnt)", value=10000)
loan_intent = st.selectbox("Tujuan Pinjaman (loan_intent)", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
loan_int_rate = st.number_input("Suku Bunga (%) (loan_int_rate)", value=10.5)
loan_percent_income = loan_amnt / (person_income + 1e-6)
cb_person_cred_hist_length = st.number_input("Lama Histori Kredit (Tahun) (cb_person_cred_hist_length)", value=3)
credit_score = st.number_input("Skor Kredit (credit_score)", min_value=300, max_value=850, value=600)
previous_loan_defaults_on_file = st.selectbox("Tunggakan Pinjaman Sebelumnya (previous_loan_defaults_on_file)", ["No", "Yes"])

# Map categorical to numeric sesuai training
gender_map = {"male": 1, "female": 0}
edu_map = {"High School": 0, "Bachelor": 1, "Master": 2, "Associate": 3, "Doctorate": 4}
default_map = {"No": 0, "Yes": 1}
home_map = {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3}
intent_map = {
    "PERSONAL": 0,
    "EDUCATION": 1,
    "MEDICAL": 2,
    "VENTURE": 3,
    "HOMEIMPROVEMENT": 4,
    "DEBTCONSOLIDATION": 5
}

# Final input dictionary
input_data = {
    "person_age": person_age,
    "person_gender": gender_map[person_gender],
    "person_education": edu_map[person_education],
    "person_income": person_income,
    "person_emp_exp": person_emp_exp,
    "person_home_ownership": home_map[person_home_ownership],
    "loan_amnt": loan_amnt,
    "loan_intent": intent_map[loan_intent],
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    "credit_score": credit_score,
    "previous_loan_defaults_on_file": default_map[previous_loan_defaults_on_file],
}

# Prediction
if st.button("🔍 Prediksi"):
    input_df = pd.DataFrame([input_data])
    result = model.predict(input_df)[0]
    st.success(f"📊 Hasil Prediksi: {'✅ DISETUJUI' if result == 1 else '❌ DITOLAK'}")

# Test Case Sidebar
st.sidebar.header("🧪 Test Case")

if st.sidebar.button("Test Case 1"):
    st.write("🔹 Laki-laki, Sarjana, Pendapatan: $60K, Pinjaman: $10K, Kredit: 720, Tujuan: PERSONAL")
    st.session_state.update({
        'person_age': 35,
        'person_gender': 'male',
        'person_education': 'Bachelor',
        'person_income': 60000,
        'person_emp_exp': 7,
        'person_home_ownership': 'OWN',
        'loan_amnt': 10000,
        'loan_intent': 'PERSONAL',
        'loan_int_rate': 12.0,
        'cb_person_cred_hist_length': 5,
        'credit_score': 720,
        'previous_loan_defaults_on_file': 'No',
    })

if st.sidebar.button("Test Case 2"):
    st.write("🔹 Perempuan, SMA, Pendapatan: $20K, Pinjaman: $15K, Kredit: 500, Tujuan: MEDICAL")
    st.session_state.update({
        'person_age': 28,
        'person_gender': 'female',
        'person_education': 'High School',
        'person_income': 20000,
        'person_emp_exp': 2,
        'person_home_ownership': 'RENT',
        'loan_amnt': 15000,
        'loan_intent': 'MEDICAL',
        'loan_int_rate': 18.5,
        'cb_person_cred_hist_length': 2,
        'credit_score': 500,
        'previous_loan_defaults_on_file': 'Yes',
    })
