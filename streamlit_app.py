# streamlit_app.py
import streamlit as st
import pandas as pd
import pickle

# Load model
@st.cache_resource
def load_model():
    with open("best_xgb_model_new.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# Load dataset (buat referensi fitur dan kategori)
@st.cache_data
def load_dataset():
    df = pd.read_csv("Dataset_A_loan.csv")
    return df

df = load_dataset()

# Ambil kategori unik untuk dropdown
education_options = sorted(df["person_education"].dropna().unique())
gender_options = sorted(df["person_gender"].dropna().unique())
home_options = sorted(df["person_home_ownership"].dropna().unique())
intent_options = sorted(df["loan_intent"].dropna().unique())
default_options = sorted(df["previous_loan_defaults_on_file"].dropna().unique())

# UI
st.title("Loan Approval Prediction")

st.subheader("Masukkan Data Calon Peminjam:")
person_age = st.number_input("Usia", min_value=18, max_value=100, value=30)
person_gender = st.selectbox("Jenis Kelamin", gender_options)
person_education = st.selectbox("Pendidikan", education_options)
person_income = st.number_input("Pendapatan Tahunan", value=50000)
person_emp_exp = st.slider("Pengalaman Kerja (tahun)", 0, 40, 5)
person_home_ownership = st.selectbox("Status Tempat Tinggal", home_options)
loan_amnt = st.number_input("Jumlah Pinjaman", value=10000)
loan_intent = st.selectbox("Tujuan Pinjaman", intent_options)
loan_int_rate = st.number_input("Suku Bunga (%)", value=10.5)
loan_percent_income = loan_amnt / (person_income + 1e-6)
cb_person_cred_hist_length = st.number_input("Lama Riwayat Kredit (tahun)", value=3)
credit_score = st.number_input("Skor Kredit", min_value=300, max_value=850, value=600)
previous_loan_defaults_on_file = st.selectbox("Riwayat Gagal Bayar", default_options)

# Gabungkan input
input_dict = {
    "person_age": person_age,
    "person_gender": person_gender,
    "person_education": person_education,
    "person_income": person_income,
    "person_emp_exp": person_emp_exp,
    "person_home_ownership": person_home_ownership,
    "loan_amnt": loan_amnt,
    "loan_intent": loan_intent,
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_cred_hist_length": cb_person_cred_hist_length,
    "credit_score": credit_score,
    "previous_loan_defaults_on_file": previous_loan_defaults_on_file,
}

input_df = pd.DataFrame([input_dict])

# Pastikan fitur sama seperti saat pelatihan (dengan one-hot encoding sesuai CSV)
X = df.drop(columns=["loan_status"])
X_encoded = pd.get_dummies(X)
input_encoded = pd.get_dummies(input_df)

# Reindex agar urutan kolom cocok
input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

# Predict
if st.button("Prediksi"):
    prediction = model.predict(input_encoded)[0]
    st.success(f"Hasil Prediksi: {'DISETUJUI' if prediction == 1 else 'DITOLAK'}")

# Tambahkan test case
st.sidebar.header("💡 Test Case")
if st.sidebar.button("Test Case 1"):
    st.write("🔹 Laki-laki, Sarjana, Pendapatan: 60K, Pinjaman: 10K, Kredit Bagus, Tujuan: PERSONAL")
if st.sidebar.button("Test Case 2"):
    st.write("🔹 Perempuan, SMA, Pendapatan: 20K, Pinjaman: 15K, Kredit Buruk, Tujuan: MEDICAL")
