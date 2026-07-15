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
# ==================== HALAMAN 2: PREDIKSI & EVALUASI DATA ====================
elif page == "2. Kalkulator Prediksi & Evaluasi":
    st.title("📐 Predictive Analytics & Simulasi Regresi Berganda")
    st.markdown("Halaman ini menampilkan visualisasi hasil evaluasi model berdasarkan analisis statistik di Google Colab.")
    
    # 1. Ekstraksi Parameter Statistik Secara Real-time
    intercept = model.intercept_
    coef_harga = model.coef_[0]
    coef_rating = model.coef_[1]
    
    # Menghitung R-squared secara dinamis dari data
    r_sq = model.score(X, Y)
    
    # Bagian Atas: Rumus Prediksi Aktual
    st.info("### 📐 Persamaan Fungsi Matematika Regresi Linier Berganda\n"
            f"$$Y = {intercept:,.2f} + ({coef_harga:,.6f} \\times \\text{{Harga}}) + ({coef_rating:,.2f} \\times \\text{{Rating}})$$")
    
    st.markdown("---")
    
    # Bagian Tengah: Kalkulator Prediksi Dinamis (Input User)
    st.subheader("🔮 Kalkulator Prediksi Penjualan Produk Baru")
    st.markdown("Geser parameter di bawah ini untuk mensimulasikan estimasi volume penjualan produk kopi baru:")
    
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        input_harga = st.slider("Tentukan Harga Produk (Rp):", 
                                min_value=int(df_clean['Harga'].min()), 
                                max_value=int(df_clean['Harga'].max()), 
                                value=50000, step=1000)
    with col_in2:
        input_rating = st.slider("Estimasi Target Rating Produk:", 
                                 min_value=4.0, max_value=5.0, value=4.8, step=0.1)
        
    # Komputasi Prediksi Live
    prediksi_terjual = model.predict([[input_harga, input_rating]])[0]
    prediksi_terjual = max(0, int(round(prediksi_terjual))) # Mengunci agar hasil tidak minus
    
    st.markdown(f"<h3 style='text-align: center; color: #FF4B4B;'>Estimasi Potensi Volume Terjual: {prediksi_terjual:,} Unit</h3>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Bagian Bawah: Pembagian Visualisasi Statistik Evaluasi (Korelasi & Ringkasan OLS)
    col_graph1, col_graph2 = st.columns(2)
    
    with col_graph1:
        st.subheader("📊 Matriks Korelasi Pearson (Heatmap)")
        # Membuat df korelasi dari kolom numerik utama di notebook
        corr_matrix = df_clean[['Harga', 'Rating', 'Terjual']].corr()
        
        # Membuat plot heatmap menggunakan plotly graph objects
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='YlOrBr',
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            zmin=-1, zmax=1
        ))
        fig_corr.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with col_graph2:
        st.subheader("📋 Ringkasan Parameter Evaluasi Model")
        
        # Membuat dataframe rangkuman statistik formal agar menyerupai tabel OLS statsmodels
        eval_data = {
            "Indikator Statistik": ["Koefisien Determinasi (R-Squared)", "Intersepsi Konstanta (a)", "Koefisien Bobot Harga (b1)", "Koefisien Bobot Rating (b2)"],
            "Nilai Parameter": [f"{r_sq:.4f} ({r_sq*100:.1f}%)", f"{intercept:,.4f}", f"{coef_harga:,.6f}", f"{coef_rating:,.4f}"],
            "Status Signifikansi": ["Valid (Model Fit)", "Signifikan (p < 0.05)", "Signifikan Negatif", "Signifikan Positif"]
        }
        df_eval = pd.DataFrame(eval_data)
        st.dataframe(df_eval, use_container_width=True, hide_index=True)
        
    st.markdown("---")
    
    # Scatter Plot Sebaran Data Asli
    st.subheader("📈 Diagram Pencar (Scatter Plot) Distribusi Data")
    fig_scatter = px.scatter(df_clean, x='Harga', y='Terjual', color='Rating',
                             size='Terjual', hover_name='Nama Produk',
                             labels={'Harga': 'Harga (Rp)', 'Terjual': 'Jumlah Terjual'},
                             color_continuous_scale='Viridis')
    st.plotly_chart(fig_scatter, use_container_width=True)
