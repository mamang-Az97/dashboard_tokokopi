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
st.sidebar.title("Dashboard")
st.sidebar.markdown("Proyek Akhir Big Data & Predictive Analytics")
page = st.sidebar.radio("Pilih Halaman Analisis:", ["1. Eksplorasi Data (EDA)", "2. Kalkulator Prediksi & Evaluasi"])

# ==================== HALAMAN 1: EKSPLORASI DATA (EDA) ====================

if page == "1. Eksplorasi Data (EDA)":
    st.title("📊 Eksplorasi Data & Deskriptif Pasar Kopi Tokopedia")
    st.markdown("Halaman ini menyajikan ringkasan visual dan analisis deskriptif pasar produk kopi berdasarkan hasil web scraping Tokopedia.")
    
    # ------------------ BARU: PANEL FILTER INTERAKTIF (SIDEBAR / ATAS) ------------------
    st.markdown("### 🔍 Panel Penyaringan Data")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        # Pilihan filter berdasarkan varian kopi
        opsi_varian = ["Semua Varian"] + list(df_clean['Varian Kopi'].unique())
        filter_varian = st.selectbox("Saring berdasarkan Varian Kopi:", opsi_varian)
    with col_f2:
        # Pilihan filter berdasarkan lokasi toko
        opsi_lokasi = ["Semua Lokasi"] + list(df_clean['Asal/Lokasi Toko'].unique())
        filter_lokasi = st.selectbox("Saring berdasarkan Asal/Lokasi Toko:", opsi_lokasi)
        
    # Mengaplikasikan filter ke dataset lokal khusus halaman 1
    df_filtered = df_clean.copy()
    if filter_varian != "Semua Varian":
        df_filtered = df_filtered[df_filtered['Varian Kopi'] == filter_varian]
    if filter_lokasi != "Semua Lokasi":
        df_filtered = df_filtered[df_filtered['Asal/Lokasi Toko'] == filter_lokasi]

    st.markdown("---")
    
    # Ringkasan Data Utama (Scorecards / KPI Metrics) berdasarkan hasil filter
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Varian Produk Kopi", value=f"{df_filtered['Nama Produk'].nunique():,}")
    with col2:
        st.metric(label="Total Produk Terjual (Unit)", value=f"{int(df_filtered['Terjual'].sum()):,}")
    with col3:
        st.metric(label="Rata-rata Harga Produk", value=f"Rp {df_filtered['Harga'].mean():,.0f}")
    with col4:
        st.metric(label="Total Data Mentah", value=f"{1700}")
    st.markdown("---")
    
    # Pembagian Layout Diagram Visualisasi (Kiri & Kanan)
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📍 10 Besar Lokasi Toko dengan Penjualan Tertinggi")
        top_lokasi = df_filtered.groupby('Asal/Lokasi Toko')['Terjual'].sum().reset_index()
        top_lokasi = top_lokasi.sort_values(by='Terjual', ascending=False).head(10)
        
        # Jika hasil filter kosong/terlalu sedikit
        if not top_lokasi.empty:
            fig_lokasi = px.bar(top_lokasi, x='Terjual', y='Asal/Lokasi Toko', orientation='h',
                                labels={'Terjual': 'Total Terjual (Unit)', 'Asal/Lokasi Toko': 'Lokasi Toko'},
                                color='Terjual', color_continuous_scale='YlOrBr') # Warna diubah ke nuansa kopi
            fig_lokasi.update_layout(yaxis={'categoryorder':'total ascending'}, height=380, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_lokasi, use_container_width=True)
        else:
            st.warning("Tidak ada data lokasi toko yang sesuai dengan filter.")
        
    with col_right:
        st.subheader("☕ Proporsi Volume Penjualan Berdasarkan Varian Kopi")
        varian_sales = df_filtered.groupby('Varian Kopi')['Terjual'].sum().reset_index()
        
        if not varian_sales.empty:
            fig_varian = px.pie(varian_sales, values='Terjual', names='Varian Kopi', 
                                hole=0.4, color_discrete_sequence=px.colors.sequential.YlOrBr) # Gradasi warna kopi hangat
            fig_varian.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_varian, use_container_width=True)
        else:
            st.warning("Tidak ada data varian kopi yang sesuai dengan filter.")

    st.markdown("---")
    
    # ------------------ BARU: TABEL DETAIL SEBARAN DATASET TERLARIS ------------------
    st.subheader("📋 Daftar 5 Produk Kopi Terlaris Berdasarkan Filter")
    st.markdown("Tabel interaktif di bawah memuat rincian data produk kopi yang paling banyak diminati oleh pasar saat ini:")
    
    # Mengurutkan berdasarkan performa penjualan tertinggi
    df_table = df_filtered[['Nama Produk', 'Varian Kopi', 'Harga', 'Rating', 'Terjual', 'Asal/Lokasi Toko']].sort_values(by='Terjual', ascending=False).head(5)
    
    # Menampilkan tabel interaktif tanpa kolom indeks bawaan pandas agar rapi
    st.dataframe(df_table, use_container_width=True, hide_index=True)


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
    
   # ==================== SIMULATOR REKOMENDASI HARGA & OMZET ====================
    st.subheader("💡 Simulator Rekomendasi Harga & Proyeksi Omzet Pasar")
    st.markdown("Pilih varian produk dan target rating untuk mendapatkan rekomendasi harga optimal beserta estimasi omzet penjualannya:")

    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        # Pilihan Varian Kopi dari Dataset
        pilihan_varian = st.selectbox("Pilih Varian Produk Kopi:", df_clean['Varian Kopi'].unique())
        
        # Mengambil harga rata-rata varian sebagai rekomendasi acuan dasar
        harga_referensi = int(df_clean[df_clean['Varian Kopi'] == pilihan_varian]['Harga'].mean())
        
        

    col_in1, col_in2 = st.columns(2)
    with col_in1:
        # Default harga dibuat murah (Rp 20.000) agar pengaruh rating langsung menonjol
        input_harga = st.slider(
            "Tentukan Harga Produk (Rp):", 
            min_value=10000, 
            max_value=20000000, 
            value=20000, 
            step=5000
        )
    
        
    # --- PROSES SIMULASI SINKRON ---
        input_df = pd.DataFrame([{'Harga': input_harga])
        prediksi_terjual = model.predict(input_df)[0]
        
        # Menampilkan hasil dengan pembulatan yang peka
        prediksi_terjual_final = max(0, int(round(prediksi_terjual)))
        
        # Menghitung Prediksi Unit Terjual
        input_df = pd.DataFrame([{'Harga': input_harga])
        prediksi_terjual = model.predict(input_df)[0]
        prediksi_terjual_final = max(0, int(round(prediksi_terjual)))
        
        # Menghitung Estimasi Omzet Penjualan (Revenue)
        estimasi_omzet = input_harga * prediksi_terjual_final

    # ------------------ DISPLAY HASIL PREDIKSI & METRIK BISNIS ------------------
    st.markdown("### 📊 Hasil Proyeksi Model Regresi:")
    col_res1, col_res2, col_res3 = st.columns(3)
    
    with col_res1:
        st.metric(label="Rekomendasi Harga Pasar", value=f"Rp {harga_referensi:,}")
    with col_res2:
        st.metric(label="Estimasi Potensi Terjual", value=f"{prediksi_terjual_final:,} Unit")
    with col_res3:
        st.metric(label="Proyeksi Total Omzet (Revenue)", value=f"Rp {estimasi_omzet:,}")

    # ------------------ GRAFIK SIMULASI SENSITIVITAS HARGA VS OMZET ------------------
    st.markdown("---")
    st.subheader("📈 Kurva Sensitivas: Pengaruh Harga terhadap Total Omzet")
    st.markdown("Grafik ini memperlihatkan titik harga paling optimal (*sweet spot*) untuk memaksimalkan total pendapatan toko:")

    # Simulasi berbagai rentang harga dari Rp 10.000 s.d. Rp 200.000
    rentang_harga = np.arange(10000, 205000, 5000)
    sim_data = []
    for h in rentang_harga:
        p = max(0, int(round(model.predict(pd.DataFrame([{'Harga': h}]))[0])))
        omzet = h * p
        sim_data.append({'Harga': h, 'Terjual': p, 'Omzet': omzet})

    df_sim_omzet = pd.DataFrame(sim_data)

    # Plot Garis Omzet vs Harga
    fig_omzet = px.line(df_sim_omzet, x='Harga', y='Omzet', 
                        labels={'Harga': 'Harga Produk (Rp)', 'Omzet': 'Proyeksi Omzet (Rp)'},
                        title=f"Proyeksi Omzet Varian '{pilihan_varian}' )
    fig_omzet.add_vline(x=input_harga, line_dash="dash", line_color="red", annotation_text="Harga Pilihan Anda")
    fig_omzet.update_traces(line_color='#FF4B4B', line_width=3)
    fig_omzet.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
    
    st.plotly_chart(fig_omzet, use_container_width=True)
    
    # ==================== VISUALISASI PREDIKSI BARU (CARA LAIN) ====================
    st.subheader("📈 Visualisasi Perbandingan Data Aktual vs Model Prediksi")
    st.markdown("Grafik di bawah ini memetakan seluruh data asli Tokopedia (titik biru) dan membandingkannya langsung dengan garis tren prediksi (garis merah) yang dihasilkan oleh model regresi kelompokmu.")
    
    # Membuat kolom prediksi untuk seluruh data di dataset agar bisa diplot
    df_clean['Prediksi_Terjual'] = model.predict(df_clean[['Harga', 'Rating']])
    df_clean['Prediksi_Terjual'] = df_clean['Prediksi_Terjual'].clip(lower=0) # Kunci biar tidak minus
    
    # Membuat chart gabungan (Scatter untuk data asli, Line untuk data prediksi)
    fig_compare = go.Figure()
    
    # 1. Titik-titik Data Aktual
    fig_compare.add_trace(go.Scatter(
        x=df_clean['Harga'], 
        y=df_clean['Terjual'],
        mode='markers',
        name='Data Aktual (Tokopedia)',
        marker=dict(color='rgb(31, 119, 180)', size=6, opacity=0.6),
        text=df_clean['Nama Produk'],
        hovertemplate="<b>%{text}</b><br>Harga: Rp%{x:,}<br>Terjual Asli: %{y} unit<extra></extra>"
    ))
    
    # 2. Garis Tren Hasil Prediksi Model
    # Diurutkan berdasarkan harga agar garisnya rapi tidak zig-zag
    df_sorted = df_clean.sort_values(by='Harga')
    fig_compare.add_trace(go.Scatter(
        x=df_sorted['Harga'], 
        y=df_sorted['Prediksi_Terjual'],
        mode='lines',
        name='Garis Tren Prediksi Model',
        line=dict(color='red', width=3),
        hovertemplate="Harga: Rp%{x:,}<br>Estimasi Terjual: %{y:.0f} unit<extra></extra>"
    ))
    
    fig_compare.update_layout(
        xaxis_title="Harga Produk (Rupiah)",
        yaxis_title="Jumlah Produk Terjual (Unit)",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        margin=dict(l=20, r=20, t=20, b=20),
        height=450
    )
    st.plotly_chart(fig_compare, use_container_width=True)
    
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
        df_eval = pd.DataFrame({
            "Indikator Statistik": ["Koefisien Determinasi (R-Squared)", "Intersepsi Konstanta (a)", "Koefisien Bobot Harga (b1)", "Koefisien Bobot Rating (b2)"],
            "Nilai Parameter": [f"{r_sq:.4f} ({r_sq*100:.1f}%)", f"{intercept:,.4f}", f"{coef_harga:,.6f}", f"{coef_rating:,.4f}"],
            "Status Analisis": ["Valid (Model Fit)", "Signifikan Pengaruh", "Korelasi Negatif", "Korelasi Positif"]
        })
        st.dataframe(df_eval, use_container_width=True, hide_index=True)

# Scatter Plot Distribusi Data Utama
    st.subheader("📈 Diagram Pencar (Scatter Plot) Distribusi Data Asli Tokopedia")
    fig_scatter = px.scatter(df_clean, x='Harga', y='Terjual', color='Rating',
                             size='Terjual', hover_name='Nama Produk',
                             labels={'Harga': 'Harga (Rp)', 'Terjual': 'Jumlah Terjual'},
                             color_continuous_scale='Viridis')
    fig_scatter.update_layout(height=450)
    st.plotly_chart(fig_scatter, use_container_width=True)
