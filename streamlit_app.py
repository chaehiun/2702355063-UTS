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

# Categorical values
person_education_list = ['Master', 'High School', 'Bachelor', 'Associate', 'Doctorate']
person_home_ownership_list = ['RENT', 'OWN', 'MORTGAGE', 'OTHER']
loan_intent_list = ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
default_options = ['No', 'Yes']
person_gender_map = {'female': 0, 'male': 1}

st.title("Loan Approval Prediction")

st.subheader("Masukkan Data Calon Peminjam")
person_age = st.number_input("Usia (person_age)", min_value=18, max_value=100, value=30)
person_gender = st.selectbox("Jenis Kelamin (person_gender)", list(person_gender_map.keys()))
person_education = st.selectbox("Pendidikan (person_education)", person_education_list)
person_income = st.number_input("Pendapatan per Tahun (person_income)", value=50000)
person_emp_exp = st.slider("Pengalaman Kerja (tahun) (person_emp_exp)", 0, 40, 5)
person_home_ownership = st.selectbox("Status Kepemilikan Tempat Tinggal (person_home_ownership)", person_home_ownership_list)
loan_amnt = st.number_input("Jumlah Pinjaman (loan_amnt)", value=10000)
loan_intent = st.selectbox("Tujuan Pinjaman (loan_intent)", loan_intent_list)
loan_int_rate = st.number_input("Suku Bunga Pinjaman (%) (loan_int_rate)", value=10.5)
loan_percent_income = loan_amnt / (person_income + 1e-6)
cb_person_cred_hist_length = st.number_input("Lama Riwayat Kredit (tahun) (cb_person_cred_hist_length)", value=3)
credit_score = st.number_input("Skor Kredit (credit_score)", min_value=300, max_value=850, value=600)
previous_loan_defaults_on_file = st.selectbox("Riwayat Gagal Bayar (previous_loan_defaults_on_file)", default_options)

# Encode input
input_data = {
    'person_age': person_age,
    'person_gender': person_gender_map[person_gender],
    'person_education': person_education_list.index(person_education),
    'person_income': person_income,
    'person_emp_exp': person_emp_exp,
    'loan_amnt': loan_amnt,
    'loan_int_rate': loan_int_rate,
    'loan_percent_income': loan_percent_income,
    'cb_person_cred_hist_length': cb_person_cred_hist_length,
    'credit_score': credit_score,
    'previous_loan_defaults_on_file': 0 if previous_loan_defaults_on_file == 'No' else 1,
}

# One-hot for intent & home ownership
for intent in loan_intent_list:
    input_data[f"loan_intent_{intent}"] = 1 if loan_intent == intent else 0
for ho in person_home_ownership_list:
    input_data[f"person_home_ownership_{ho}"] = 1 if person_home_ownership == ho else 0

# Predict
if st.button("Prediksi"):
    input_df = pd.DataFrame([input_data])
    result = model.predict(input_df)[0]
    st.success(f"Hasil Prediksi: {'DISETUJUI' if result == 1 else 'DITOLAK'}")

# Test Cases
st.sidebar.header("💡 Test Cases")
if st.sidebar.button("Test Case 1"):
    st.write("Laki-laki, Sarjana, Pendapatan 60000, Pinjaman 10000, Kredit 700, Tujuan: PERSONAL")
    st.write("Prediksi manual: Jalankan dengan input yang sesuai di form utama")

if st.sidebar.button("Test Case 2"):
    st.write("Perempuan, SMA, Pendapatan 20000, Pinjaman 15000, Kredit 500, Tujuan: MEDICAL")
    st.write("Prediksi manual: Jalankan dengan input yang sesuai di form utama")
