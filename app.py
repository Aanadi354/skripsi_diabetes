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
]

df_summary = pd.DataFrame(data_summary)
# ================= CONFUSION MATRIX SUMMARY =================
cm_data = [
    {
        "Skenario": "AdaBoost (Ori)",
        "TN": 232, "FP": 5, "FN": 3, "TP": 252,
        "Accuracy": 0.9746, "Presisi": 0.9765, "Recall": 0.9881, "F1": 0.9822
    },
    {
        "Skenario": "AdaBoost + ADASYN",
        "TN": 222, "FP": 15, "FN": 12, "TP": 240,
        "Accuracy": 0.9407, "Presisi": 0.9639, "Recall": 0.9524, "F1": 0.9581
    },
    {
        "Skenario": "XGBoost (Ori)",
        "TN": 235, "FP": 2, "FN": 0, "TP": 255,
        "Accuracy": 0.9831, "Presisi": 0.9767, "Recall": 1.0000, "F1": 0.9882
    },
    {
        "Skenario": "XGBoost + ADASYN",
        "TN": 232, "FP": 5, "FN": 3, "TP": 252,
        "Accuracy": 0.9746, "Presisi": 0.9765, "Recall": 0.9881, "F1": 0.9822
    }
]

df_cm = pd.DataFrame(cm_data)


# ======================= SIDEBAR MENU =======================
menu = st.sidebar.selectbox(
    "Navigasi",
    [
        "Home",
        "Dataset",
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
elif menu == "Dataset":
    st.header("Dataset Penelitian")

    tab1, tab2, tab3, tab4 = st.tabs(["Dataset Original", "Penjelasan Fitur", "Dataset Hasil Normalisasi", "Hasil Oversampling"])

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
        st.subheader("Dataset Hasil Normalisasi")
        try:
            df_norm = pd.read_csv("data_preprocessed_normalized.csv")
            st.dataframe(df_norm, use_container_width=True)
            st.write("Jumlah data:", df_norm.shape)
        except Exception as e:
            st.error(f"Terjadi error saat memuat dataset normalisasi: {e}")

    with tab4:
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
    st.header("Dashboard Evaluasi Model")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "AdaBoost (Original)",
        "AdaBoost + ADASYN",
        "XGBoost (Original)",
        "XGBoost + ADASYN",
        "Rangkuman Hasil Skenario Uji Coba"
    ])

    # ================= TAB 1 =================
    with tab1:
        st.subheader("Confusion Matrix AdaBoost (Original)")
        st.image("cm_adaboost_ori.png", use_container_width=True)

        st.markdown("""
        ### 📊 Interpretasi Hasil
        - Accuracy : **97.46%**
        - Recall : **98.81%**
        - Precision : **97.65%**
        - F1 Score : **98.22%**

        Model AdaBoost tanpa oversampling sudah memiliki performa sangat baik.
        Tingkat recall tinggi menunjukkan model mampu mendeteksi hampir semua
        kasus diabetes dengan benar.
        """)

    # ================= TAB 2 =================
    with tab2:
        st.subheader("Confusion Matrix AdaBoost + ADASYN")
        st.image("cm_adaboost_adasyn.png", use_container_width=True)

        st.markdown("""
        ### 📊 Interpretasi Hasil
        - Accuracy : **94.07%**
        - Recall : **95.24%**
        - Precision : **96.39%**
        - F1 Score : **95.81%**

        Setelah penerapan ADASYN, distribusi data menjadi lebih seimbang.
        Namun accuracy sedikit menurun karena adanya data sintetis.
        """)

    # ================= TAB 3 =================
    with tab3:
        st.subheader("Confusion Matrix XGBoost (Original)")
        st.image("cm_xgboost_ori.png", use_container_width=True)

        st.markdown("""
        ### 📊 Interpretasi Hasil
        - Accuracy : **98.31%**
        - Recall : **100%**
        - Precision : **97.67%**
        - F1 Score : **98.82%**

        XGBoost original menghasilkan performa terbaik.
        Recall mencapai 100% artinya semua pasien diabetes berhasil terdeteksi.
        """)

    # ================= TAB 4 =================
    with tab4:
        st.subheader("Confusion Matrix XGBoost + ADASYN")
        st.image("cm_xgboost_adasyn.png", use_container_width=True)

        st.markdown("""
        ### 📊 Interpretasi Hasil
        - Accuracy : **97.46%**
        - Recall : **98.81%**
        - Precision : **97.65%**
        - F1 Score : **98.22%**

        Oversampling ADASYN tidak meningkatkan performa XGBoost secara signifikan.
        Hal ini menunjukkan model sudah mampu menangani imbalance data dengan baik.
        """)
        # ================= TAB 5 =================
    with tab5:
        st.subheader("Rangkuman Hasil Seluruh Skenario")

        st.write("### 📊 Tabel Confusion Matrix & Performa")

        # Highlight best accuracy
        best_acc = df_cm["Accuracy"].max()

        def highlight_best(row):
            color = "background-color: yellow" if row["Accuracy"] == best_acc else ""
            return [color]*len(row)

        st.dataframe(
            df_cm.style.apply(highlight_best, axis=1),
            use_container_width=True
        )

        st.caption("🟨 Warna kuning menunjukkan model dengan performa terbaik")

        # ================= BAGAN PERBANDINGAN =================
        st.write("### 📈 Grafik Perbandingan Kinerja")

        metrics = ["Accuracy", "Recall", "F1"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for i, metric in enumerate(metrics):
            axes[i].bar(df_cm["Skenario"], df_cm[metric])
            axes[i].set_title(metric)
            axes[i].set_ylim(0.9, 1.01)
            axes[i].tick_params(axis='x', rotation=45)

        st.pyplot(fig)

        # ================= BAGAN CONFUSION MATRIX =================
        # st.write("### 📊 Perbandingan Nilai Confusion Matrix")

        # cm_values = ["TP", "TN", "FP", "FN"]

        # fig2, ax = plt.subplots(figsize=(10,6))

        # for val in cm_values:
        #     ax.plot(df_cm["Skenario"], df_cm[val], marker="o", label=val)

        # ax.legend()
        # ax.set_title("Perbandingan Nilai Confusion Matrix")
        # ax.tick_params(axis='x', rotation=45)

        # st.pyplot(fig2)



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
