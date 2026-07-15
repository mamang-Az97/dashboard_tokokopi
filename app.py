import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# 1. Konfigurasi Halaman & Tema Konten
st.set_page_config(page_title="Dashboard Analisis Prediktif Kopi Tokopedia", layout="wide")

# 2. Memuat Dataset Bersih
@st.cache_data
def load_data():
    # Pastikan file CSV ini berada di folder yang sama dengan app.py
    df = pd.read_csv("data_kopi_tokopedia_clean.csv")
    return df

try:
    df_clean = load_data()
except FileNotFoundError:
    st.error("File 'data_kopi_tokopedia_clean.csv' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    st.stop()

# 3. Membuat Model Regresi di Latar Belakang (Untuk Prediksi)
X = df_clean[['Harga', 'Rating']]
Y = df_clean['Terjual']
model = LinearRegression()
model.fit(X, Y)

# 4. Navigasi Halaman (Sidebar Menu)
st.sidebar.title("Navigasi Dashboard")
page = st.sidebar.radio("Pilih Halaman:", ["1. Eksplorasi Data (EDA)", "2. Kalkulator Prediksi & Evaluasi"])

# ==================== HALAMAN 1: EDA ====================
if page == "1. Eksplorasi Data (EDA)":
    st.title("📊 Eksplorasi Data & Deskriptif Pasar Kopi Tokopedia")
    st.markdown("Halaman ini menyajikan ringkasan visual pasar produk kopi berdasarkan hasil web scraping.")
    
    # Kpi Metrics (Scorecards)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Total Varian Produk Kopi", value=f"{df_clean['Nama Produk'].nunique():,}")
    with col2:
        st.metric(label="Total Produk Terjual (Unit)", value=f"{int(df_clean['Terjual'].sum()):,}")
    with col3:
        st.metric(label="Rata-rata Harga Produk", value=f"Rp {df_clean['Harga'].mean():,.0f}")
        
    st.markdown("---")
    
    # Pembagian Diagram Visualisasi (Dua Kolom)
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("10 Besar Lokasi Toko dengan Penjualan Tertinggi")
        top_lokasi = df_clean.groupby('Asal/Lokasi Toko')['Terjual'].sum().reset_index()
        top_lokasi = top_lokasi.sort_values(by='Terjual', ascending=False).head(10)
        fig_lokasi = px.bar(top_lokasi, x='Terjual', y='Asal/Lokasi Toko', orientation='h',
                            labels={'Terjual': 'Total Terjual', 'Asal/Lokasi Toko': 'Lokasi Toko'},
                            color='Terjual', color_continuous_scale='Viridis')
        fig_lokasi.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_lokasi, use_container_width=True)
        
    with col_right:
        st.subheader("Proporsi Penjualan Berdasarkan Varian Kopi")
        varian_sales = df_clean.groupby('Varian Kopi')['Terjual'].sum().reset_index()
        fig_varian = px.pie(varian_sales, values='Terjual', names='Varian Kopi', 
                    hole=0.4, color_discrete_sequence=px.colors.sequential.Blues)
        st.plotly_chart(fig_varian, use_container_width=True)

# ==================== HALAMAN 2: PREDIKSI ====================
elif page == "2. Kalkulator Prediksi & Evaluasi":
    st.title("📐 Predictive Analytics & Simulasi Regresi Berganda")
    st.markdown("Halaman ini menampilkan visualisasi hasil evaluasi model dan kalkulator prediksi dinamis.")
    
    # Bagian Atas: Rumus & Metrik Evaluasi Statis
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.info("### Persamaan Regresi Linier Berganda\n"
                f"$$Y = {model.intercept_:.2f} + ({model.coef_[0]:.6f} \\times \\text{{Harga}}) + ({model.coef_[1]:.2f} \\times \\text{{Rating}})$$")
    with col_b:
        # Masukkan angka R-Squared riil dari hasil Colab kelompokmu di sini
        st.success("### R-Squared Model\n**34.5%** dari variasi penjualan dijelaskan oleh Harga & Rating.")
        
    st.markdown("---")
    
    # Bagian Tengah: Simulasi Prediksi Interaktif (Input User)
    st.subheader("🔮 Kalkulator Prediksi Penjualan Produk Baru")
    st.markdown("Sesuaikan parameter di bawah ini untuk mengestimasi potensi volume penjualan produk kopi baru:")
    
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        input_harga = st.slider("Tentukan Harga Produk (Rp):", 
                                min_value=int(df_clean['Harga'].min()), 
                                max_value=int(df_clean['Harga'].max()), 
                                value=50000, step=1000)
    with col_in2:
        input_rating = st.slider("Estimasi Target Rating Produk:", 
                                 min_value=4.0, max_value=5.0, value=4.8, step=0.1)
        
    # Komputasi Prediksi Secara Live
    prediksi_terjual = model.predict([[input_harga, input_rating]])[0]
    prediksi_terjual = max(0, int(round(prediksi_terjual))) # Hasil tidak boleh negatif
    
    # Tampilkan Hasil Prediksi
    st.markdown(f"<h3 style='text-align: center; color: #FF4B4B;'>Estimasi Potensi Penjualan: {prediksi_terjual:,} Unit Produk</h3>", unsafe_allowed_index=True)
    
    st.markdown("---")
    
    # Bagian Bawah: Scatter Plot Interaktif Tren Regresi
    st.subheader("📈 Scatter Plot Tren Distribusi Data")
    fig_scatter = px.scatter(df_clean, x='Harga', y='Terjual', color='Rating',
                             size='Terjual', hover_name='Nama Produk',
                             labels={'Harga': 'Harga (Rp)', 'Terjual': 'Jumlah Terjual'},
                             title="Hubungan Antara Harga vs Kuantitas Terjual (Warna Berdasarkan Rating)")
    st.plotly_chart(fig_scatter, use_container_width=True)
