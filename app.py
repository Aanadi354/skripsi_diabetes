import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Diabetes Mellitus Diagnostic App", layout="wide")

# ======================= LOAD ASSETS =======================
@st.cache_resource
def load_assets():
    model = joblib.load('best_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

model, scaler = load_assets()

# ======================= SUMMARY DATA =======================
data_summary = [
    {'Skenario': 'Skenario 1', 'Model': 'AdaBoost (Ori)', 'Accuracy': 0.9746, 'CV Accuracy': 0.9660, 'Precision': 0.9765, 'Recall': 0.9881, 'F1-Score': 0.9822},
    {'Skenario': 'Skenario 2', 'Model': 'AdaBoost + ADASYN', 'Accuracy': 0.9407, 'CV Accuracy': 0.9304, 'Precision': 0.9639, 'Recall': 0.9524, 'F1-Score': 0.9581},
    {'Skenario': 'Skenario 3', 'Model': 'XGBoost (Ori)', 'Accuracy': 0.9831, 'CV Accuracy': 0.9779, 'Precision': 0.9767, 'Recall': 1.0000, 'F1-Score': 0.9882},
    {'Skenario': 'Skenario 4', 'Model': 'XGBoost + ADASYN', 'Accuracy': 0.9746, 'CV Accuracy': 0.9405, 'Precision': 0.9765, 'Recall': 0.9881, 'F1-Score': 0.9822},
    {'Skenario': 'Skenario 5', 'Model': 'AdaBoost + Tuning', 'Accuracy': 0.9661, 'CV Accuracy': 0.9389, 'Precision': 0.9651, 'Recall': 0.9881, 'F1-Score': 0.9765},
]

df_summary = pd.DataFrame(data_summary)

# ======================= SIDEBAR MENU =======================
menu = st.sidebar.selectbox("Navigasi", [
    "Home",
    "Data",
    "Dashboard Performa Model",
    "Prediksi Diagnosis"
])

# ======================= HOME =======================
if menu == "Home":
    st.title("ANALISIS PERBANDINGAN KINERJA ADABOOST DAN XGBOOST DENGAN PENERAPAN OVERSAMPLING ADASYN PADA KLASIFIKASI DIABETES")

    st.image("diabetes.jpg", use_column_width=False, width=250)

    st.subheader("Latar Belakang")
    st.write("""
    Diabetes Mellitus merupakan salah satu penyakit kronis yang jumlah penderitanya terus meningkat setiap tahun di dunia maupun di Indonesia. 
    Deteksi dini sangat penting untuk mencegah komplikasi serius seperti penyakit jantung, gagal ginjal, dan kerusakan saraf. 
    Namun pada praktiknya, diagnosis sering terlambat karena pemeriksaan membutuhkan waktu dan interpretasi medis yang kompleks. 
    Oleh karena itu, penelitian ini memanfaatkan teknik machine learning untuk membantu klasifikasi diagnosis diabetes secara otomatis.
    Metode yang dibandingkan adalah AdaBoost dan XGBoost dengan penerapan teknik penyeimbangan data ADASYN untuk meningkatkan performa model klasifikasi.
    Sistem ini diharapkan dapat menjadi alat bantu keputusan (decision support system) bagi tenaga kesehatan dalam melakukan deteksi awal penyakit diabetes.
    """)

# ======================= DATASET =======================
elif menu == "Data":
    st.header("Dataset Penelitian")
    try:
        df = pd.read_excel("DATASET_SKRIPSI_AAN.xlsx", engine="openpyxl")
        st.success("Dataset berhasil dimuat")
        st.dataframe(df, use_container_width=True)
        st.write("Jumlah data:", df.shape)

    except Exception as e:
        st.error(f"Terjadi error: {e}")


# ======================= DASHBOARD =======================
elif menu == "Dashboard Performa Model":
    st.header("Ringkasan Performa Model (Skenario 1-5)")
    st.dataframe(df_summary.style.highlight_max(axis=0, subset=['Accuracy', 'Recall', 'F1-Score']))

    metrics = ['Accuracy', 'Recall', 'F1-Score']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, metric in enumerate(metrics):
        sns.barplot(x='Skenario', y=metric, data=df_summary, ax=axes[i], palette='magma')
        axes[i].set_title(f'Perbandingan {metric}', fontweight='bold')
        axes[i].set_ylim(0.85, 1.02)
        for p in axes[i].patches:
            axes[i].annotate(format(p.get_height(), '.4f'),
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    st.pyplot(fig)

# ======================= PREDICTION =======================
else:
    st.header("Form Input Data Pasien")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            umur = st.number_input("Umur", min_value=1, max_value=120, value=50)
            jk = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            imt = st.number_input("IMT (Indeks Massa Tubuh)", min_value=10.0, max_value=60.0, value=25.0)
            lingkar_perut = st.number_input("Lingkar Perut (cm)", min_value=50, max_value=200, value=85)

        with col2:
            sistolik = st.number_input("Tekanan Darah Sistolik", min_value=70, max_value=250, value=120)
            diastolik = st.number_input("Tekanan Darah Diastolik", min_value=40, max_value=150, value=80)
            hba1c = st.number_input("Hba1c (%)", min_value=3.0, max_value=20.0, value=6.5)
            gd_puasa = st.number_input("Gula Darah Puasa (GDPuasa)", min_value=50.0, max_value=600.0, value=110.0)
            gd_2pp = st.number_input("Gula Darah 2 Jam PP (GD2PP)", min_value=50.0, max_value=600.0, value=140.0)

        submit_button = st.form_submit_button("Prediksi Diagnosis")

    if submit_button:
        jk_val = 0 if jk == "Laki-laki" else 1

        input_data = pd.DataFrame([[umur, jk_val, imt, lingkar_perut, sistolik, diastolik, hba1c, gd_puasa, gd_2pp]],
                                  columns=['Umur', 'JK', 'IMT', 'Lingkar Perut', 'Sistolik', 'Diastolik', 'Hba1c', 'GDPuasa', 'GD2PP'])

        num_cols = ['Umur', 'IMT', 'Lingkar Perut', 'Sistolik', 'Diastolik', 'Hba1c', 'GDPuasa', 'GD2PP']
        input_data[num_cols] = scaler.transform(input_data[num_cols])

        prediction = model.predict(input_data)[0]

        st.subheader("Hasil Prediksi:")
        if prediction == 1:
            st.error("Pasien Terdiagnosa: DIABETES MELLITUS (DM)")
        else:
            st.success("Pasien Terdiagnosa: NON DIABETES MELLITUS (NON-DM)")
