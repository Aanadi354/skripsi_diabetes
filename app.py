import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Diabetes Mellitus Diagnostic App",
    layout="wide"
)

# ======================= LOAD MODEL & SCALER =======================
@st.cache_resource
def load_assets():
    model = joblib.load("best_model.joblib")
    scaler = joblib.load("scaler.joblib")
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
menu = st.sidebar.selectbox(
    "Navigasi",
    [
        "Home",
        "Data",
        "Dashboard Performa Model",
        "Prediksi Diagnosis"
    ]
)

# ======================= HOME =======================
if menu == "Home":
    st.title(
        "ANALISIS PERBANDINGAN KINERJA ADABOOST DAN XGBOOST "
        "DENGAN PENERAPAN OVERSAMPLING ADASYN PADA KLASIFIKASI DIABETES"
    )

    st.image("diabetes.jpg", width=250)

    st.subheader("Latar Belakang")
    st.write("""
    Diabetes Mellitus merupakan salah satu penyakit kronis dengan jumlah penderita
    yang terus meningkat. Penelitian ini membandingkan kinerja AdaBoost dan XGBoost
    dengan penerapan oversampling ADASYN untuk meningkatkan performa klasifikasi.
    """)

# ======================= DATASET =======================
elif menu == "Data":
    st.header("Dataset Penelitian")

    tab1, tab2, tab3 = st.tabs(["📄 Dataset Original", "📚 Penjelasan Fitur", "🧪 Hasil Oversampling"])

    with tab1:
        try:
            df = pd.read_csv("DATASET_SKRIPSI_AAN.csv", sep=None, engine="python")
            st.success("Dataset berhasil dimuat")
            st.dataframe(df, use_container_width=True)
            st.write("Jumlah data:", df.shape)

            if "DX" in df.columns:
                class_counts = df["DX"].value_counts()

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Distribusi Kelas")
                    st.dataframe(class_counts.rename("Jumlah"))

                with col2:
                    fig, ax = plt.subplots()
                    class_counts.plot(kind="bar", ax=ax)
                    ax.set_title("Diabetes vs Non-Diabetes")
                    st.pyplot(fig)
            else:
                st.warning("Kolom DX tidak ditemukan")

        except Exception as e:
            st.error(e)

    with tab2:
        st.subheader("Penjelasan Fitur Dataset")
        fitur_df = pd.DataFrame({
            "Fitur": [
                "Umur", "JK", "IMT", "Lingkar Perut", "Sistolik", "Diastolik",
                "HbA1c", "GDPuasa", "GD2PP", "DX"
            ],
            "Keterangan": [
                "Usia pasien dalam tahun",
                "Jenis kelamin (0=Laki-laki, 1=Perempuan)",
                "Indeks Massa Tubuh (kg/m²)",
                "Lingkar perut (cm) indikator obesitas sentral",
                "Tekanan darah sistolik (mmHg)",
                "Tekanan darah diastolik (mmHg)",
                "Kadar HbA1c (%) rata-rata gula darah 3 bulan",
                "Glukosa darah puasa (mg/dL)",
                "Glukosa darah 2 jam setelah makan (mg/dL)",
                "Label diagnosis (0=Non Diabetes, 1=Diabetes)"
            ]
        })
        st.dataframe(fitur_df, use_container_width=True)
        st.info("Tab ini membantu pengguna memahami arti setiap variabel medis sebelum melihat hasil model.")

    with tab3:
        st.subheader("Dataset Hasil Oversampling (ADASYN)")

        try:
            df_over = pd.read_csv("hasil_adasyn_oversampling.csv")
            st.success("Dataset oversampling berhasil dimuat")

            def highlight_oversampled(row):
                return [
                    "background-color: #fff3cd" if row["is_oversampled"] else ""
                    for _ in row
                ]

            st.dataframe(
                df_over.style.apply(highlight_oversampled, axis=1),
                use_container_width=True
            )

            st.write("Jumlah data:", df_over.shape)

            if "DX" in df_over.columns:
                st.subheader("Distribusi Kelas Setelah Oversampling")

                class_counts = df_over["DX"].value_counts().sort_index()
                class_counts.index = class_counts.index.map({0: "Non Diabetes (0)", 1: "Diabetes (1)"})

                col1, col2 = st.columns(2)

                with col1:
                    st.write("Jumlah data tiap kelas:")
                    st.dataframe(class_counts.rename("Jumlah"))

                with col2:
                    fig, ax = plt.subplots()
                    class_counts.plot(kind="bar", ax=ax)
                    ax.set_xlabel("Kelas")
                    ax.set_ylabel("Jumlah Data")
                    ax.set_title("Distribusi Diabetes vs Non-Diabetes Setelah ADASYN")
                    for i, v in enumerate(class_counts.values):
                        ax.text(i, v + (0.01 * max(class_counts.values)), str(v), ha="center")
                    st.pyplot(fig)

            else:
                st.warning("Kolom 'DX' tidak ditemukan pada dataset oversampling")

            st.caption("🟨 Baris berwarna kuning menandakan data hasil oversampling ADASYN")

        except Exception as e:
            st.error(f"Terjadi error: {e}")

# ======================= DASHBOARD =======================
elif menu == "Dashboard Performa Model":
    st.header("Ringkasan Performa Model")

    st.dataframe(
        df_summary.style.highlight_max(
            axis=0,
            subset=["Accuracy", "Recall", "F1-Score"]
        ),
        use_container_width=True
    )

    metrics = ["Accuracy", "Recall", "F1-Score"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, metric in enumerate(metrics):
        sns.barplot(
            x="Skenario",
            y=metric,
            data=df_summary,
            ax=axes[i],
            palette="magma"
        )
        axes[i].set_ylim(0.85, 1.02)
        axes[i].set_title(metric)

    st.pyplot(fig)

# ======================= PREDIKSI =======================
else:
    st.header("Form Input Data Pasien")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            umur = st.number_input("Umur", 1, 120, 50)
            jk = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
            imt = st.number_input("IMT", 10.0, 60.0, 25.0)
            lingkar_perut = st.number_input("Lingkar Perut", 50, 200, 85)

        with col2:
            sistolik = st.number_input("Sistolik", 70, 250, 120)
            diastolik = st.number_input("Diastolik", 40, 150, 80)
            hba1c = st.number_input("HbA1c", 3.0, 20.0, 6.5)
            gd_puasa = st.number_input("GDPuasa", 50.0, 600.0, 110.0)
            gd_2pp = st.number_input("GD2PP", 50.0, 600.0, 140.0)

        submit = st.form_submit_button("Prediksi")

    if submit:
        jk_val = 0 if jk == "Laki-laki" else 1

        input_df = pd.DataFrame([[umur, jk_val, imt, lingkar_perut, sistolik, diastolik, hba1c, gd_puasa, gd_2pp]],
                                columns=["Umur", "JK", "IMT", "Lingkar Perut", "Sistolik", "Diastolik", "Hba1c", "GDPuasa", "GD2PP"])

        num_cols = input_df.columns.drop("JK")
        input_df[num_cols] = scaler.transform(input_df[num_cols])

        pred = model.predict(input_df)[0]

        if pred == 1:
            st.error("DIABETES MELLITUS")
        else:
            st.success("NON DIABETES MELLITUS")
