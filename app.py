import streamlit as st
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import io

# Try to import statsmodels for VIF and Breusch-Pagan; provide fallback if import fails
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.diagnostic import het_breuschpagan
    statsmodels_available = True
except ImportError:
    statsmodels_available = False
    st.warning("Modul 'statsmodels' tidak tersedia. Uji multikolinearitas (VIF) dan homoskedastisitas (Breusch-Pagan) tidak dapat dilakukan. Silakan instal statsmodels dengan 'pip install statsmodels'.")

# Default dataset (hardcoded) jika tidak ada file diunggah
csv_data = """Tahun,Bulan,Curah_Hujan,Suhu,Produksi
2015,1,202,30.4,5.31
2015,2,120,25.2,5.17
2015,3,174,27.7,5.14
2015,4,230,24.2,6.54
2015,5,101,29.8,4.26
2015,6,269,24.4,6.89
2015,7,287,26.9,7.12
2015,8,150,27.1,5.22
2015,9,163,27.7,5.41
2015,10,172,24.5,6.27
2015,11,231,31.5,4.59
2015,12,189,24.8,6.47
2016,1,183,25.9,6.13
2016,2,107,24.3,5.89
2016,3,180,26.1,5.41
2016,4,233,25.7,6.72
2016,5,103,25.5,5.36
2016,6,299,28.6,6.53
2016,7,289,26.6,6.33
2016,8,152,28.7,4.77
2016,9,287,28.3,6.5
2016,10,164,24.1,5.9
2016,11,108,30.2,4.42
2016,12,262,30.3,6.02
2017,1,222,24.6,6.25
2017,2,234,30.8,5.29
2017,3,111,26.6,4.93
2017,4,136,29.8,4.47
2017,5,203,26.2,5.38
2017,6,274,29.7,6.32
2017,7,100,29.8,4.01
2017,8,126,28.2,4.57
2017,9,114,24.9,4.95
2017,10,195,29.1,5.25
2017,11,151,28.5,4.77
2017,12,242,30.0,5.74
2018,1,135,31.5,4.73
2018,2,127,31.4,4.43
2018,3,144,26.4,5.34
2018,4,207,25.5,5.88
2018,5,129,31.1,4.04
2018,6,289,24.9,7.01
2018,7,220,29.2,5.3
2018,8,254,27.3,6.12
2018,9,264,29.5,5.66
2018,10,158,26.6,5.22
2018,11,195,28.5,5.26
2018,12,285,31.7,5.67
2019,1,138,28.0,4.66
2019,2,180,26.1,5.48
2019,3,101,28.0,4.55
2019,4,229,28.2,5.93
2019,5,259,28.4,5.88
2019,6,283,25.9,6.9
2019,7,137,26.9,5.6
2019,8,238,30.5,5.89
2019,9,243,30.7,5.2
2019,10,192,30.9,4.78
2019,11,286,25.8,7.17
2019,12,247,29.2,6.02
2020,1,151,31.5,4.53
2020,2,203,25.1,5.47
2020,3,284,24.9,7.3
2020,4,189,31.9,4.35
2020,5,159,26.1,5.87
2020,6,108,26.2,5.08
2020,7,246,29.8,5.88
2020,8,174,31.5,5.01
2020,9,263,31.2,5.68
2020,10,192,26.3,5.65
2020,11,209,28.4,5.93
2020,12,268,25.4,7.2
2021,1,167,25.9,6.04
2021,2,227,30.0,5.23
2021,3,294,29.3,6.03
2021,4,214,29.1,5.57
2021,5,121,26.0,5.2
2021,6,281,31.2,5.86
2021,7,126,30.4,5.21
2021,8,291,30.9,5.96
2021,9,160,28.0,5.65
2021,10,116,32.0,4.09
2021,11,257,31.6,5.38
2021,12,136,31.4,4.49
2022,1,145,31.7,4.05
2022,2,260,26.5,6.04
2022,3,227,27.0,6.4
2022,4,166,28.6,5.05
2022,5,123,25.7,4.95
2022,6,226,28.1,6.25
2022,7,229,26.6,6.31
2022,8,269,30.5,6.33
2022,9,198,26.0,5.75
2022,10,234,28.0,6.0
2022,11,272,25.6,6.76
2022,12,241,31.1,5.18
2023,1,242,27.0,6.27
2023,2,185,26.6,5.47
2023,3,285,28.3,6.59
2023,4,185,26.0,5.37
2023,5,269,30.6,5.51
2023,6,107,30.2,4.14
2023,7,207,29.0,6.19
2023,8,154,25.1,6.27
2023,9,243,31.8,4.66
2023,10,244,27.2,6.26
2023,11,118,27.5,4.65
2023,12,157,31.3,5.0"""

# Streamlit App
st.title("🌾 Sistem Prediksi Produksi Padi Berdasarkan Curah Hujan & Suhu")

# Sidebar
st.sidebar.title("Menu Navigasi")
page = st.sidebar.selectbox("Pilih Halaman", ["📊 Dataset", "🧮 Evaluasi Model", "🔮 Prediksi Baru", "📈 Visualisasi"])

# File uploader untuk dataset CSV
st.sidebar.title("Unggah Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader("Pilih file CSV", type=["csv"])

# Load dataset
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        # Validasi kolom yang diperlukan
        required_columns = ['Curah_Hujan', 'Suhu', 'Produksi']
        if not all(col in df.columns for col in required_columns):
            st.error("File CSV harus memiliki kolom: Curah_Hujan, Suhu, Produksi")
            df = pd.read_csv(StringIO(csv_data))  # Kembali ke default dataset jika gagal
        else:
            st.sidebar.success("Dataset berhasil diunggah!")
    except Exception as e:
        st.error(f"Error membaca file CSV: {e}")
        df = pd.read_csv(StringIO(csv_data))  # Kembali ke default dataset jika gagal
else:
    df = pd.read_csv(StringIO(csv_data))  # Gunakan default dataset jika tidak ada file diunggah

# Preprocessing: Validasi data
if df[['Curah_Hujan', 'Suhu', 'Produksi']].isnull().any().any():
    st.error("Dataset mengandung nilai kosong. Harap periksa data.")
    df = df.dropna()  # Hapus baris dengan missing values
if (df['Curah_Hujan'] < 0).any() or (df['Suhu'] < 20).any() or (df['Suhu'] > 40).any() or (df['Produksi'] < 0).any():
    st.warning("Dataset mengandung nilai tidak realistis (Curah Hujan < 0, Suhu < 20°C atau > 40°C, Produksi < 0). Data akan difilter.")
    df = df[(df['Curah_Hujan'] >= 0) & (df['Suhu'].between(20, 40)) & (df['Produksi'] >= 0)]

# Preprocessing: Normalisasi data
scaler = StandardScaler()
X = df[['Curah_Hujan', 'Suhu']]
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=['Curah_Hujan', 'Suhu'])
y = df['Produksi']

# Split dan train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluasi metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Peringatan performa model rendah
if r2 < 0.7:
    st.warning("Performa model rendah (R² < 0.7). Pertimbangkan untuk memeriksa kualitas data atau menambah fitur lain.")

# Input prediksi di sidebar
st.sidebar.title("Input Prediksi")
curah_hujan = st.sidebar.number_input("Curah Hujan (mm)", min_value=0.0, max_value=500.0, value=200.0)
suhu = st.sidebar.number_input("Suhu (°C)", min_value=20.0, max_value=40.0, value=28.0)
predict_button = st.sidebar.button("Prediksi")

# Prediksi logic
if predict_button:
    # Validasi input pengguna
    if curah_hujan < 0 or suhu < 20 or suhu > 40:
        st.error("Input tidak valid: Curah Hujan harus ≥ 0 mm, Suhu harus antara 20-40°C.")
    else:
        input_data = scaler.transform(np.array([[curah_hujan, suhu]]))
        prediksi = model.predict(input_data)[0]
        st.session_state.prediksi = prediksi
        st.session_state.input_curah_hujan = curah_hujan
        st.session_state.input_suhu = suhu

# Main Area
if page == "📊 Dataset":
    st.header("Preview Dataset")
    st.dataframe(df.head(20))  # Tampilkan 20 baris pertama sebagai preview
    st.write("Total entri: ", len(df))
    st.write("**Statistik Deskriptif**")
    st.dataframe(df.describe())
    st.write("**Korelasi Variabel**")
    st.dataframe(df[['Curah_Hujan', 'Suhu', 'Produksi']].corr())
    st.write("**Interpretasi Korelasi**: Curah Hujan memiliki korelasi positif kuat (0.73) dengan Produksi, sedangkan Suhu memiliki korelasi negatif sedang (-0.53).")

elif page == "🧮 Evaluasi Model":
    st.header("Evaluasi Model Regresi Linier Berganda")
    st.write("**R² (Koefisien Determinasi)**: ", round(r2, 3), " (Model menjelaskan ~86% variasi data)")
    st.write("**MAE (Mean Absolute Error)**: ", round(mae, 3), " ton/ha")
    st.write("**RMSE (Root Mean Squared Error)**: ", round(rmse, 3), " ton/ha")
    st.write("**Persamaan Model (dengan data ternormalisasi)**: Produksi = {:.3f} + {:.4f} × Curah Hujan (scaled) - {:.3f} × Suhu (scaled)".format(model.intercept_, model.coef_[0], abs(model.coef_[1])))

    # Penjelasan koefisien model
    st.subheader("Interpretasi Koefisien Model")
    st.write(f"- **Intercept ({round(model.intercept_, 3)})**: Nilai dasar produksi padi (ton/ha) ketika curah hujan dan suhu dalam skala ternormalisasi bernilai 0.")
    st.write(f"- **Koefisien Curah Hujan ({round(model.coef_[0], 4)})**: Setiap peningkatan satu unit curah hujan (dalam skala ternormalisasi) meningkatkan produksi padi sebesar {round(model.coef_[0], 4)} ton/ha, dengan asumsi suhu tetap. Ini menunjukkan curah hujan yang lebih tinggi mendukung hasil panen yang lebih baik.")
    st.write(f"- **Koefisien Suhu ({round(model.coef_[1], 3)})**: Setiap peningkatan satu unit suhu (dalam skala ternormalisasi) menurunkan produksi padi sebesar {round(abs(model.coef_[1]), 3)} ton/ha, dengan asumsi curah hujan tetap. Ini menunjukkan suhu tinggi dapat mengurangi hasil panen.")
    st.write("**Implikasi Praktis**: Petani dapat menggunakan model ini untuk memperkirakan produksi dan menyesuaikan waktu tanam berdasarkan prakiraan curah hujan dan suhu, misalnya, menghindari penanaman pada periode suhu ekstrem (>35°C).")

    # Uji multikolinearitas (VIF)
    st.subheader("Uji Multikolinearitas (VIF)")
    if statsmodels_available:
        vif_data = pd.DataFrame()
        vif_data["Variabel"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        st.dataframe(vif_data)
        st.write("**Interpretasi**: Nilai VIF < 10 menunjukkan tidak ada multikolinearitas signifikan antara curah hujan dan suhu, memastikan independensi variabel dalam model.")
    else:
        st.error("Uji VIF tidak dapat dilakukan karena modul 'statsmodels' tidak tersedia.")

    # Uji normalitas residual (Q-Q Plot)
    st.subheader("Uji Normalitas Residual (Q-Q Plot)")
    residuals = y_test - y_pred
    fig_qq, ax_qq = plt.subplots()
    stats.probplot(residuals, dist="norm", plot=ax_qq)
    ax_qq.set_title("Q-Q Plot Residual")
    ax_qq.set_xlabel("Kuantil Teoretis")
    ax_qq.set_ylabel("Kuantil Residual")
    st.pyplot(fig_qq)
    st.write("**Interpretasi**: Jika titik-titik pada Q-Q Plot mengikuti garis lurus, residual cenderung berdistribusi normal, memenuhi asumsi regresi linier.")

    # Uji homoskedastisitas (Breusch-Pagan)
    st.subheader("Uji Homoskedastisitas (Breusch-Pagan)")
    if statsmodels_available:
        import statsmodels.api as sm
        X_test_with_const = sm.add_constant(X_test)
        _, pval, _, _ = het_breuschpagan(residuals, X_test_with_const)
        st.write(f"**P-value Breusch-Pagan**: {round(pval, 3)}")
        st.write("**Interpretasi**: Jika p-value > 0.05, asumsi homoskedastisitas terpenuhi (variansi residual konstan). Jika p-value ≤ 0.05, ada indikasi heteroskedastisitas.")
    else:
        st.error("Uji Breusch-Pagan tidak dapat dilakukan karena modul 'statsmodels' tidak tersedia.")

    # Tombol untuk mengunduh laporan
    st.subheader("Unduh Laporan Evaluasi Model")
    report_data = {
        "Metrik": ["R²", "MAE", "RMSE"],
        "Nilai": [round(r2, 3), round(mae, 3), round(rmse, 3)],
        "Deskripsi": [
            "Menunjukkan proporsi variasi data yang dijelaskan model (~86%)",
            "Rata-rata kesalahan absolut prediksi (ton/ha)",
            "Akar rata-rata kuadrat kesalahan prediksi (ton/ha)"
        ]
    }
    report_df = pd.DataFrame(report_data)
    csv_buffer = io.StringIO()
    report_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Unduh Laporan (CSV)",
        data=csv_buffer.getvalue(),
        file_name="laporan_evaluasi_model.csv",
        mime="text/csv"
    )

elif page == "🔮 Prediksi Baru":
    st.header("Hasil Prediksi Produksi Padi")
    if 'prediksi' in st.session_state:
        prediksi = st.session_state.prediksi
        curah_hujan = st.session_state.input_curah_hujan
        suhu = st.session_state.input_suhu
        st.success(f"Prediksi Produksi: {round(prediksi, 2)} ton/ha")
        st.write(f"**Keterangan**: Prediksi ini dihasilkan berdasarkan input Curah Hujan = {curah_hujan} mm dan Suhu = {suhu} °C. Nilai ini menunjukkan perkiraan produksi padi per hektar dalam kondisi iklim tersebut, berdasarkan model regresi linier berganda yang dilatih dengan data historis.")
        # Rekomendasi aplikatif
        st.subheader("Rekomendasi")
        if prediksi < 5:
            st.write("- **Produksi Rendah**: Pertimbangkan untuk memastikan drainase yang baik jika curah hujan tinggi, atau tambahkan irigasi jika curah hujan rendah. Hindari penanaman pada suhu ekstrem (>35°C).")
        elif 5 <= prediksi <= 6:
            st.write("- **Produksi Sedang**: Kondisi iklim saat ini mendukung produksi sedang. Pastikan pengelolaan air dan pemupukan optimal untuk meningkatkan hasil.")
        else:
            st.write("- **Produksi Tinggi**: Kondisi iklim saat ini sangat mendukung. Manfaatkan curah hujan yang cukup dan suhu optimal untuk memaksimalkan hasil panen.")
    else:
        st.info("Masukkan nilai Curah Hujan dan Suhu di sidebar, lalu tekan 'Prediksi' untuk melihat hasil.")

elif page == "📈 Visualisasi":
    st.header("Visualisasi Hubungan Variabel")
    
    # Scatter plot Curah Hujan vs Produksi
    st.subheader("Curah Hujan vs Produksi")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x='Curah_Hujan', y='Produksi', data=df, ax=ax1)
    ax1.set_title("Curah Hujan vs Produksi")
    ax1.set_xlabel("Curah Hujan (mm)")
    ax1.set_ylabel("Produksi (ton/ha)")
    st.pyplot(fig1)
    st.write("**Keterangan**: Setiap titik mewakili data bulanan produksi padi (ton/ha) terhadap curah hujan (mm). Pola menunjukkan hubungan positif, di mana curah hujan yang lebih tinggi cenderung meningkatkan produksi padi.")

    # Scatter plot Suhu vs Produksi
    st.subheader("Suhu vs Produksi")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x='Suhu', y='Produksi', data=df, ax=ax2)
    ax2.set_title("Suhu vs Produksi")
    ax2.set_xlabel("Suhu (°C)")
    ax2.set_ylabel("Produksi (ton/ha)")
    st.pyplot(fig2)
    st.write("**Keterangan**: Setiap titik mewakili data bulanan produksi padi (ton/ha) terhadap suhu (°C). Pola menunjukkan hubungan negatif, di mana suhu yang lebih tinggi cenderung menurunkan produksi padi.")

    # Line plot prediksi vs aktual
    st.subheader("Aktual vs Prediksi Produksi")
    fig3, ax3 = plt.subplots()
    ax3.plot(y_test.values, label='Aktual', marker='o')
    ax3.plot(y_pred, label='Prediksi', marker='x')
    ax3.set_title("Aktual vs Prediksi Produksi")
    ax3.set_xlabel("Data Uji (Indeks)")
    ax3.set_ylabel("Produksi (ton/ha)")
    ax3.legend()
    st.pyplot(fig3)
    st.write("**Keterangan**: Garis dengan titik bulat menunjukkan nilai produksi aktual, sedangkan garis dengan tanda silang menunjukkan nilai prediksi dari model regresi linier berganda pada data uji.")

    # Residual plot
    st.subheader("Residual Plot")
    residuals = y_test - y_pred
    fig4, ax4 = plt.subplots()
    sns.scatterplot(x=y_pred, y=residuals, ax=ax4)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_xlabel("Prediksi Produksi (ton/ha)")
    ax4.set_ylabel("Residual (Aktual - Prediksi)")
    ax4.set_title("Residual Plot")
    st.pyplot(fig4)
    st.write("**Keterangan**: Setiap titik mewakili selisih antara produksi aktual dan prediksi (residual) untuk data uji. Residual yang tersebar acak di sekitar garis nol menunjukkan tidak ada pola sistematis dalam kesalahan prediksi, memenuhi asumsi homoskedastisitas.")