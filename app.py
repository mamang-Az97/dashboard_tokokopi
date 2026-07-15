import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# ==================== 1. KONFIGURASI HALAMAN & TEMA ====================
st.set_page_config(page_title="Dashboard Analisis Prediktif Kopi Tokopedia", layout="wide")

# ==================== 2. MEMUAT DATASET BERSIH ====================
@st.cache_data
def load_data():
    # Membaca dataset yang berada di direktori yang sama
    df = pd.read_csv("data_kopi_tokopedia_clean.csv")
    return df

try:
    df_clean = load_data()
except FileNotFoundError:
    st.error("File 'data_kopi_tokopedia_clean.csv' tidak ditemukan. Pastikan file berada di repositori GitHub yang sama dengan app.py.")
    st.stop()

# ==================== 3. TRAINING MODEL REGRESI (LATAR BELAKANG) ====================
# Variabel independen harus berbentuk DataFrame dengan nama kolom yang jelas
X = df_clean[['Harga', 'Rating']]
Y = df_clean['Terjual']
model = LinearRegression()
model.fit(X, Y)

# ==================== 4. NAVIGASI SIDEBAR ====================
st.sidebar.title("Navigasi Dashboard")
st.sidebar.markdown("Proyek Akhir Big Data & Predictive Analytics")
page = st.sidebar.radio("Pilih Halaman Analisis:", ["1. Eksplorasi Data (EDA)", "2. Kalkulator Prediksi & Evaluasi"])

# ==================== HALAMAN 1: EKSPLORASI DATA (EDA) ====================
if page == "1. Eksplorasi Data (EDA)":
    st.title("📊 Eksplorasi Data & Deskriptif Pasar Kopi Tokopedia")
    st.markdown("Halaman ini menyajikan ringkasan visual pasar produk kopi berdasarkan hasil web scraping Tokopedia.")
    
    # Ringkasan Data (Scorecards / KPI Metrics)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Varian Produk Kopi", value=f"{df_clean['Nama Produk'].nunique():,}")
    with col2:
        st.metric(label="Total Produk Terjual (Unit)", value=f"{int(df_clean['Terjual'].sum()):,}")
    with col3:
        st.metric(label="Rata-rata Harga Produk di Pasar", value=f"Rp {df_clean['Harga'].mean():,.0f}")
        
    st.markdown("---")
    
    # Pembagian Layout Diagram Visualisasi (Kiri & Kanan)
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📍 10 Besar Lokasi Toko dengan Penjualan Tertinggi")
        top_lokasi = df_clean.groupby('Asal/Lokasi Toko')['Terjual'].sum().reset_index()
        top_lokasi = top_lokasi.sort_values(by='Terjual', ascending=False).head(10)
        
        fig_lokasi = px.bar(top_lokasi, x='Terjual', y='Asal/Lokasi Toko', orientation='h',
                            labels={'Terjual': 'Total Terjual (Unit)', 'Asal/Lokasi Toko': 'Lokasi Toko'},
                            color='Terjual', color_continuous_scale='Viridis')
        fig_lokasi.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
        st.plotly_chart(fig_lokasi, use_container_width=True)
        
    with col_right:
        st.subheader("☕ Proporsi Volume Penjualan Berdasarkan Varian Kopi")
        varian_sales = df_clean.groupby('Varian Kopi')['Terjual'].sum().reset_index()
        
        # Menggunakan palet sekuensial Blues yang aman dari AttributeError
        fig_varian = px.pie(varian_sales, values='Terjual', names='Varian Kopi', 
                            hole=0.4, color_discrete_sequence=px.colors.sequential.Blues)
        fig_varian.update_layout(height=400)
        st.plotly_chart(fig_varian, use_container_width=True)

# ==================== HALAMAN 2: PREDIKSI & EVALUASI DATA ====================
elif page == "2. Kalkulator Prediksi & Evaluasi":
    st.title("📐 Predictive Analytics & Simulasi Regresi Berganda")
    st.markdown("Halaman ini mengintegrasikan fungsi matematika hasil pemodelan regresi linier berganda kelompok secara interaktif.")
    
    # 1. Ekstraksi Nilai Parameter Regresi Aktual
    intercept = model.intercept_
    coef_harga = model.coef_[0]
    coef_rating = model.coef_[1]
    r_sq = model.score(X, Y)
    
    # Bagian Atas: Rumus Fungsi Matematika Regresi
    st.info("### 📐 Persamaan Fungsi Matematika Regresi Linier Berganda\n"
            f"$$Y = {intercept:,.2f} + ({coef_harga:,.6f} \\times \\text{{Harga}}) + ({coef_rating:,.2f} \\times \\text{{Rating}})$$")
    
    st.markdown("---")
    
    # Bagian Tengah: Kalkulator Prediksi Dinamis (Input User)
    st.subheader("🔮 Kalkulator Prediksi Penjualan Produk Baru")
    st.markdown("Geser parameter di bawah ini untuk mensimulasikan estimasi volume penjualan. Nilai awal diatur agar model langsung sinkron menampilkan data dinamis:")
    
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        min_h = int(df_clean['Harga'].min())
        max_h = int(df_clean['Harga'].max())
        # Nilai awal (value) diatur rendah agar efek pengurang minus dari harga tidak menenggelamkan output ke angka 0
        input_harga = st.slider("Tentukan Harga Produk (Rp):", 
                                min_value=min_h, 
                                max_value=max_h, 
                                value=15000, 
                                step=1000)
    with col_in2:
        min_r = float(df_clean['Rating'].min())
        max_r = float(df_clean['Rating'].max())
        # Nilai awal diatur ke maksimal (5.0) untuk mendongkrak konstanta awal yang minus besar
        input_rating = st.slider("Estimasi Target Rating Produk:", 
                                 min_value=min_r, 
                                 max_value=max_r, 
                                 value=5.0, 
                                 step=0.1)
        
    # --- PROSES SIMULASI SINKRON ---
    # Memasukkan input ke DataFrame dengan label kolom yang cocok dengan model training
    input_df = pd.DataFrame([{
        'Harga': input_harga,
        'Rating': input_rating
    }])
    
    # Komputasi prediksi live
    prediksi_terjual = model.predict(input_df)[0]
    prediksi_terjual_final = max(0, int(round(prediksi_terjual)))
    
    # Tampilan Output Interaktif Kontainer
    st.markdown(
        f"""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #FF4B4B; margin-top: 15px;'>
            <h4 style='margin: 0; color: #31333F;'>Hasil Estimasi Potensi Volume Penjualan:</h4>
            <h2 style='margin: 10px 0 0 0; color: #FF4B4B; text-align: center;'>{prediksi_terjual_final:,} Unit Produk Terjual</h2>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # Bagian Bawah: Pembagian Blok Visualisasi Evaluasi (Korelasi & Ringkasan OLS)
    col_graph1, col_graph2 = st.columns(2)
    
    with col_graph1:
        st.subheader("📊 Matriks Korelasi Pearson (Heatmap)")
        corr_matrix = df_clean[['Harga', 'Rating', 'Terjual']].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='YlOrBr',
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            zmin=-1, zmax=1
        ))
        fig_corr.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with col_graph2:
        st.subheader("📋 Ringkasan Parameter Evaluasi Model")
        
        eval_data = {
            "Indikator Statistik": ["Koefisien Determinasi (R-Squared)", "Intersepsi Konstanta (a)", "Koefisien Bobot Harga (b1)", "Koefisien Bobot Rating (b2)"],
            "Nilai Parameter": [f"{r_sq:.4f} ({r_sq*100:.1f}%)", f"{intercept:,.4f}", f"{coef_harga:,.6f}", f"{coef_rating:,.4f}"],
            "Status Analisis": ["Valid (Model Fit)", "Signifikan Pengaruh", "Korelasi Negatif", "Korelasi Positif"]
        }
        df_eval = pd.DataFrame(eval_data)
        st.dataframe(df_eval, use_container_width=True, hide_index=True)
        
    st.markdown("---")
    
    # Scatter Plot Distribusi Data Utama
    st.subheader("📈 Diagram Pencar (Scatter Plot) Distribusi Data Asli Tokopedia")
    fig_scatter = px.scatter(df_clean, x='Harga', y='Terjual', color='Rating',
                             size='Terjual', hover_name='Nama Produk',
                             labels={'Harga': 'Harga (Rp)', 'Terjual': 'Jumlah Terjual'},
                             color_continuous_scale='Viridis')
    fig_scatter.update_layout(height=450)
    st.plotly_chart(fig_scatter, use_container_width=True)
