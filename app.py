import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Konfigurasi Halaman
# ─────────────────────────────────────────────
st.set_page_config(
    page_title='Prediksi Produksi Padi — Sumatera',
    page_icon='🌾',
    layout='wide',
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, label {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
[data-testid="stAppViewContainer"] { background-color: #F8FAFC; }
.block-container { padding-top: 2.5rem !important; padding-bottom: 2rem !important; }
[data-testid="stSidebar"] { background-color: #0F172A !important; }
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label { color: #F1F5F9 !important; font-weight: 600 !important; }
.main-title {
    background: linear-gradient(135deg, #166534 0%, #16A34A 60%, #4ADE80 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
}
.sub-title { font-size: 15px; color: #64748B; margin-bottom: 30px; font-weight: 500; }
.hero-box {
    background: linear-gradient(135deg, #14532D 0%, #15803D 100%);
    color: white; padding: 35px; border-radius: 24px;
    box-shadow: 0 20px 25px -5px rgba(21,128,61,0.2);
    text-align: center; margin-bottom: 28px;
}
.hero-value {
    font-size: 46px; font-weight: 800; color: #FFFFFF;
    margin-top: 8px; letter-spacing: -0.5px;
}
.premium-card {
    background: #FFFFFF; padding: 24px; border-radius: 18px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.04); border: 1px solid #E2E8F0;
    margin-bottom: 20px;
}
.grid-badge {
    background: #F0FDF4; padding: 14px; border-radius: 12px;
    border-left: 4px solid #16A34A; margin-bottom: 10px;
    font-size: 14px;
}
.welcome-card {
    background: #FFFFFF; padding: 40px; border-radius: 24px;
    border: 1px solid #E2E8F0; border-top: 8px solid #16A34A;
    box-shadow: 0 10px 25px rgba(0,0,0,0.05);
    text-align: center; max-width: 800px; margin: 40px auto;
}
div.stButton > button:first-child {
    background: linear-gradient(135deg, #16A34A 0%, #15803D 100%) !important;
    color: white !important; border-radius: 12px !important;
    border: none !important; font-weight: 700 !important;
    font-size: 15px !important; padding: 14px 24px !important;
    box-shadow: 0 4px 14px rgba(22,163,74,0.35) !important;
    transition: all 0.25s ease !important;
}
div.stButton > button:first-child:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 25px rgba(22,163,74,0.5) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load Artefak
# ─────────────────────────────────────────────
@st.cache_resource
def load_artefak():
    model  = joblib.load('regresi_berganda.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_csv('data_clean.csv')

model, scaler = load_artefak()
df = load_data()

FITUR_SCALE = ['Luas Panen', 'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
PROVINSI_LIST = sorted([
    'Sumatera Utara', 'Sumatera Barat', 'Riau',
    'Jambi', 'Sumatera Selatan', 'Bengkulu', 'Lampung'
])

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown(
    '<div style="font-size:40px;font-weight:800;margin-bottom:5px;">'
    '🌾 <span class="main-title">Prediksi Produksi Padi — Sumatera</span></div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-title">Sistem Prediksi Produksi Padi Berbasis '
    'Multiple Linear Regression | Data 1993–2020</div>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────
# Sidebar Input
# ─────────────────────────────────────────────
st.sidebar.markdown(
    "<h2 style='color:#F8FAFC;font-size:20px;font-weight:700;margin-bottom:18px;'>"
    "⚙️ Parameter Input</h2>",
    unsafe_allow_html=True
)

provinsi = st.sidebar.selectbox(
    '🗺️ Provinsi',
    options=PROVINSI_LIST,
    help='Pilih provinsi di wilayah Sumatera'
)

luas_panen = st.sidebar.number_input(
    '🌱 Luas Panen (Ha)',
    min_value=1_000,
    max_value=2_000_000,
    value=300_000,
    step=1_000,
    format='%d'
)

curah_hujan = st.sidebar.number_input(
    '🌧️ Curah Hujan (mm/tahun)',
    min_value=500,
    max_value=5_000,
    value=1_800,
    step=50,
    format='%d'
)

kelembapan = st.sidebar.number_input(
    '💧 Kelembapan (%)',
    min_value=50.0,
    max_value=100.0,
    value=82.0,
    step=0.5,
    format='%.1f'
)

suhu = st.sidebar.number_input(
    '🌡️ Suhu Rata-rata (°C)',
    min_value=18.0,
    max_value=38.0,
    value=27.0,
    step=0.1,
    format='%.1f'
)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
tombol = st.sidebar.button('🚀 Prediksi Produksi', use_container_width=True)

# ─────────────────────────────────────────────
# Tab
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(['🔮 Prediksi', '📊 Transparansi Model', '📈 Eksplorasi Data'])

# ── TAB 1: PREDIKSI ──────────────────────────
with tab1:
    if tombol:
        try:
            X_input = pd.DataFrame([[luas_panen, curah_hujan, kelembapan, suhu]],
                                   columns=FITUR_SCALE)
            X_sc   = scaler.transform(X_input)
            hasil  = model.predict(X_sc)[0]

            if hasil < 0:
                st.error('⚠️ Prediksi menghasilkan nilai negatif. '
                         'Periksa kembali nilai input — kemungkinan di luar rentang data latih.')
            else:
                st.markdown(f"""
                <div class="hero-box">
                    <div style="font-size:14px;font-weight:600;text-transform:uppercase;
                                letter-spacing:2px;color:#86EFAC;">
                        Estimasi Produksi Padi
                    </div>
                    <div class="hero-value">{hasil:,.0f} Ton</div>
                    <div style="font-size:13px;color:#BBF7D0;margin-top:8px;font-style:italic;">
                        Provinsi {provinsi} · Prediksi berdasarkan model OLS
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                st.markdown(
                    "<h3 style='margin-top:0;font-size:18px;color:#14532D;font-weight:700;'>"
                    "📋 Ringkasan Parameter Input</h3>",
                    unsafe_allow_html=True
                )
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"<div class='grid-badge'><b>🗺️ Provinsi:</b><br>{provinsi}</div>",
                                unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<div class='grid-badge'><b>🌱 Luas Panen:</b><br>{luas_panen:,} Ha</div>",
                                unsafe_allow_html=True)
                with c3:
                    st.markdown(f"<div class='grid-badge'><b>🌧️ Curah Hujan:</b><br>{curah_hujan:,} mm</div>",
                                unsafe_allow_html=True)
                    st.markdown(f"<div class='grid-badge'><b>💧 Kelembapan:</b><br>{kelembapan:.1f} %</div>",
                                unsafe_allow_html=True)
                with c4:
                    st.markdown(f"<div class='grid-badge'><b>🌡️ Suhu Rata-rata:</b><br>{suhu:.1f} °C</div>",
                                unsafe_allow_html=True)

                # Konteks vs data historis
                median_prod = df['Produksi'].median()
                selisih = hasil - median_prod
                pct = (selisih / median_prod) * 100
                arah = '🔼 di atas' if selisih > 0 else '🔽 di bawah'
                st.markdown(
                    f"<br><div style='font-size:13px;color:#64748B;'>"
                    f"Prediksi {arah} median historis "
                    f"({median_prod:,.0f} ton) sebesar <b>{abs(pct):.1f}%</b></div>",
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f'Terjadi error: {e}')
    else:
        st.markdown(f"""
        <div class="welcome-card">
            <div style="font-size:48px;margin-bottom:12px;">🌾</div>
            <h3 style="color:#0F172A;font-weight:800;margin-top:0;">
                Selamat Datang di Sistem Prediksi Produksi Padi
            </h3>
            <p style="color:#64748B;font-size:14px;line-height:1.7;
                      max-width:560px;margin:0 auto 20px;">
                Masukkan parameter pertanian (luas panen, curah hujan, kelembapan,
                suhu) di panel kiri untuk mendapatkan estimasi produksi padi
                di wilayah Sumatera.
            </p>
            <div style="display:inline-block;background:#F0FDF4;
                        padding:10px 22px;border-radius:30px;
                        font-weight:600;color:#16A34A;font-size:13px;">
                Isi parameter di sidebar kiri untuk memulai
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── TAB 2: TRANSPARANSI MODEL ─────────────────
with tab2:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown(
        "<h3 style='margin-top:0;color:#0F172A;font-weight:700;'>"
        "📊 Koefisien Model (Multiple Linear Regression)</h3>",
        unsafe_allow_html=True
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        df_koef = pd.DataFrame({
            'Fitur': FITUR_SCALE,
            'Koefisien (β)': [f'{b:+,.2f}' for b in model.coef_],
            'Arah Pengaruh': ['⬆️ Positif' if b > 0 else '⬇️ Negatif' for b in model.coef_]
        })
        st.dataframe(df_koef, use_container_width=True, hide_index=True)

    with col_b:
        st.metric('Intercept (β₀)', f'{model.intercept_:,.2f}')
        st.markdown("""
        **Interpretasi Koefisien:**
        - **Luas Panen** → pengaruh terbesar terhadap produksi
        - **Curah Hujan** → berpengaruh positif
        - **Kelembapan** → berpengaruh negatif (kelembapan berlebih menurunkan produksi)
        - **Suhu** → berpengaruh negatif (suhu terlalu tinggi menurunkan hasil)

        *Koefisien dalam skala terstandarisasi (StandardScaler)*
        """)
    st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 3: EKSPLORASI DATA ────────────────────
with tab3:
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown(
        "<h3 style='margin-top:0;color:#0F172A;font-weight:700;'>"
        "📈 Data Historis Produksi Padi Sumatera (1993–2020)</h3>",
        unsafe_allow_html=True
    )

    # Rekonstruksi kolom Provinsi dari one-hot
    df_viz = df.copy()
    prov_cols = [c for c in df.columns if c.startswith('Provinsi_')]
    df_viz['Provinsi'] = 'Sumatera Utara'
    for col in prov_cols:
        nama = col.replace('Provinsi_', '')
        df_viz.loc[df_viz[col] == True, 'Provinsi'] = nama

    # Filter provinsi
    prov_filter = st.multiselect(
        'Filter Provinsi',
        options=PROVINSI_LIST,
        default=PROVINSI_LIST[:3]
    )

    if prov_filter:
        df_filtered = df_viz[df_viz['Provinsi'].isin(prov_filter)]
    else:
        df_filtered = df_viz

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric('Total Data', f'{len(df_filtered):,} baris')
    with col_m2:
        st.metric('Rata-rata Produksi', f'{df_filtered["Produksi"].mean():,.0f} ton')
    with col_m3:
        st.metric('Produksi Tertinggi', f'{df_filtered["Produksi"].max():,.0f} ton')

    st.dataframe(
        df_filtered[['Tahun', 'Provinsi', 'Produksi', 'Luas Panen',
                     'Curah hujan', 'Kelembapan', 'Suhu rata-rata']
                   ].sort_values(['Provinsi', 'Tahun']).reset_index(drop=True),
        use_container_width=True,
        height=380
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────
st.markdown("<br><hr style='border-top:1px solid #E2E8F0;'>", unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    st.markdown("<div style='color:#94A3B8;font-size:12px;'>© 2026 · Prediksi Produksi Padi Sumatera</div>",
                unsafe_allow_html=True)
with c2:
    st.markdown("<div style='text-align:right;color:#94A3B8;font-size:12px;'>"
                "Model: Multiple Linear Regression · Scaler: StandardScaler</div>",
                unsafe_allow_html=True)
