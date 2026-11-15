import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import os
import requests
from bs4 import BeautifulSoup
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Wasabil Corefine AI-Cyber Cognitive Defense System | Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS untuk Tampilan SOC Modern & Dark Glassmorphism
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
    }
    .stMetric {
        background: linear-gradient(135deg, #131b2e 0%, #1a243d 100%);
        border: 1px solid #2a3b5c;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .stMetric label {
        color: #8fa6cb !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }
    .stMetric div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.9rem !important;
        font-weight: 700 !important;
    }
    .header-banner {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border-left: 5px solid #38bdf8;
        padding: 20px 25px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .keyword-badge {
        display: inline-block;
        background-color: #1e293b;
        color: #38bdf8;
        border: 1px solid #38bdf8;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin-right: 6px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Model & Dataset
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('model_cyber_bilstm.keras')
    with open('tokenizer_cyber.json', 'r', encoding='utf-8') as f:
        tokenizer = tokenizer_from_json(f.read())
    return model, tokenizer

@st.cache_data
def load_dataset():
    for fname in ['dataset_cyber_final.csv', 'dataset_cyber_alerts_LATEST.csv', 'dataset_cyber_alerts.csv']:
        if os.path.exists(fname):
            df = pd.read_csv(fname)
            return df, fname
    return None, None

model, tokenizer = load_assets()
df_raw, loaded_filename = load_dataset()

LABEL_MAP = {
    0: "🟢 AMAN / NETRAL",
    1: "🟡 POTENSI KONFLIK",
    2: "🔴 PROPAGANDA / DISINFORMASI"
}

LABEL_DESC = {
    0: "Berita informatif reguler tanpa indikasi provokasi atau disinformasi.",
    1: "Mengandung narasi polarisasi, kerusuhan, atau potensi gesekan sosial.",
    2: "Terindikasi kuat sebagai narasi provokatif, klaim palsu, atau disinformasi."
}

# 4. Fungsi Utilitas NLP & Scraper
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_single(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=50, padding='post', truncating='post')
    probs = model.predict(padded, verbose=0)[0]
    idx = int(np.argmax(probs))

    # Ekstrak kata yang dikenali oleh kamus AI
    word_index = tokenizer.word_index
    recognized_words = [w for w in cleaned.split() if w in word_index and word_index[w] < 5000]
    return idx, probs, cleaned, recognized_words

def extract_title_from_url(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    og_title = soup.find("meta", property="og:title")
    judul = og_title["content"].strip() if (og_title and og_title.get("content")) else (soup.title.string.strip() if soup.title else None)
    if judul:
        for sep in [" - ", " | "]:
            if sep in judul:
                judul = judul.split(sep)[0].strip()
    return judul

# 5. Sidebar Filter
with st.sidebar:
    st.title("🛡️ Wasabil Corefine AI")
    st.caption("Engine: Bidirectional LSTM v2.2")
    st.divider()

    df_filtered = df_raw.copy() if df_raw is not None else None
    if df_filtered is not None:
        st.subheader("🎛️ Filter Pemantauan")
        if 'keyword' in df_filtered.columns:
            all_kw = ["Semua Topik"] + sorted(df_filtered['keyword'].dropna().unique().tolist())
            selected_kw = st.selectbox("Filter Kata Kunci:", all_kw)
            if selected_kw != "Semua Topik":
                df_filtered = df_filtered[df_filtered['keyword'] == selected_kw]

        label_col = next((c for c in ['sentiment', 'label', 'kelas', 'kategori'] if c in df_filtered.columns), None)
        if label_col:
            df_filtered['Kategori_Nama'] = df_filtered[label_col].map(LABEL_MAP).fillna(df_filtered[label_col].astype(str))

        st.info(f"Menampilkan **{len(df_filtered):,}** artikel dari total **{len(df_raw):,}** data.")

# 6. Banner Utama
st.markdown("""
<div class="header-banner">
    <h2 style="margin:0; color:#f8fafc;">🛡️ Wasabil Corefine AI – Cyber Cognitive Defense System Threat & Disinformation Intelligence</h2>
    <p style="margin:5px 0 0 0; color:#94a3b8;">Sistem Pemantauan Isu Siber & Klasifikasi Berita Real-Time Berbasis Deep Learning (BiLSTM)</p>
</div>
""", unsafe_allow_html=True)

# 7. Navigasi Utama (Tab Terpisah Agar Lebih Bersih)
menu_utama = st.tabs(["🔍 Uji Deteksi Real-Time", "📊 Dashboard Intelijen Data", "📂 Analisis Massal (CSV)"])

# === TAB 1: UJI DETEKSI REAL-TIME ===
with menu_utama[0]:
    col_in, col_out = st.columns([1.1, 1.2], gap="large")

    with col_in:
        st.subheader("Input Target Analisis")
        mode_input = st.radio("Pilih Metode Input:", ["📝 Teks Judul / Klaim", "🔗 Tautan (URL) Berita"], horizontal=True)

        teks_uji = ""
        if mode_input == "📝 Teks Judul / Klaim":
            txt = st.text_area("Masukkan judul berita:", height=130, placeholder="Contoh: Sebarkan! Beredar kabar hoaks ribuan pasukan asing...")
            if st.button("🚀 Jalankan Analisis AI", type="primary", use_container_width=True):
                if txt.strip():
                    teks_uji = txt.strip()
                else:
                    st.warning("Masukkan teks terlebih dahulu.")
        else:
            url_in = st.text_input("Tempelkan URL portal berita:", placeholder="https://detik.com...")
            if st.button("🌐 Tarik Judul & Analisis", type="primary", use_container_width=True):
                if url_in.strip():
                    with st.spinner("Mengambil metadata artikel..."):
                        try:
                            judul_url = extract_title_from_url(url_in.strip())
                            if judul_url:
                                st.success(f"**Judul Ditemukan:** {judul_url}")
                                teks_uji = judul_url
                            else:
                                st.error("Judul tidak ditemukan di halaman tersebut.")
                        except Exception as e:
                            st.error(f"Situs memblokir akses otomatis. Gunakan mode Teks Manual. ({e})")
                else:
                    st.warning("Masukkan link URL terlebih dahulu.")

    with col_out:
        st.subheader("Hasil Diagnosis Model")
        if teks_uji:
            idx, probs, cleaned, tokens = predict_single(teks_uji)
            conf = float(probs[idx] * 100)

            # Tampilan Kartu Status
            if idx == 0:
                st.success(f"### {LABEL_MAP[idx]}\n{LABEL_DESC[idx]}")
            elif idx == 1:
                st.warning(f"### {LABEL_MAP[idx]}\n{LABEL_DESC[idx]}")
            else:
                st.error(f"### {LABEL_MAP[idx]}\n{LABEL_DESC[idx]}")

            # Gauge Chart Tingkat Keyakinan
            gauge_color = "#22c55e" if idx == 0 else ("#eab308" if idx == 1 else "#ef4444")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf,
                number={'suffix': "%", 'font': {'size': 32, 'color': 'white'}},
                title={'text': "Tingkat Keyakinan AI", 'font': {'size': 14, 'color': '#94a3b8'}},
                gauge={
                    'axis': {'range': (0, 100), 'tickcolor': "white"},
                    'bar': {'color': gauge_color},
                    'bgcolor': "#1e293b",
                    'steps': [
                        {'range': (0, 50), 'color': '#0f172a'},
                        {'range': (50, 75), 'color': '#1e293b'},
                        {'range': (75, 100), 'color': '#334155'}
                    ]
                }

            ))

            fig_gauge.update_layout(height=210, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Rincian Probabilitas & Token Kata
            st.write("**Distribusi Probabilitas Kelas:**")
            prob_df = pd.DataFrame({
                "Kategori": [LABEL_MAP[0], LABEL_MAP[1], LABEL_MAP[2]],
                "Skor (%)": [round(float(p)*100, 2) for p in probs]
            })
            st.dataframe(prob_df, hide_index=True, use_container_width=True)

            if tokens:
                st.write("**Kata Kunci yang Dibaca AI:**")
                badges = "".join([f'<span class="keyword-badge">{w}</span>' for w in tokens])
                st.markdown(badges, unsafe_allow_html=True)
        else:
            st.info("👈 Masukkan judul berita atau tautan URL di panel sebelah kiri untuk melihat skor ancaman, grafik gauge, dan analisis kata.")

# === TAB 2: DASHBOARD INTELIJEN DATA ===
with menu_utama[1]:
    if df_filtered is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Artikel", f"{len(df_filtered):,}")
        m2.metric("Sumber Media", f"{df_filtered['source'].nunique():,}" if 'source' in df_filtered.columns else "-")
        m3.metric("Kata Kunci Aktif", f"{df_filtered['keyword'].nunique()}" if 'keyword' in df_filtered.columns else "-")
        if 'Kategori_Nama' in df_filtered.columns:
            propaganda_cnt = (df_filtered['Kategori_Nama'] == LABEL_MAP[2]).sum()
            m4.metric("Terdeteksi Propaganda", f"{propaganda_cnt:,}")
        else:
            m4.metric("Arsitektur Model", "BiLSTM")

        st.divider()
        c_left, c_right = st.columns(2)

        with c_left:
            if 'Kategori_Nama' in df_filtered.columns:
                color_map = {
                    LABEL_MAP[0]: "#22c55e",
                    LABEL_MAP[1]: "#eab308",
                    LABEL_MAP[2]: "#ef4444"
                }
                fig_pie = px.pie(
                    df_filtered,
                    names='Kategori_Nama',
                    title="Proporsi Kategori Ancaman Berita",
                    hole=0.5,
                    color='Kategori_Nama',
                    color_discrete_map=color_map
                )
                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig_pie, use_container_width=True)

        with c_right:
            if 'keyword' in df_filtered.columns:
                top_kw = df_filtered['keyword'].value_counts().head(10).reset_index()
                top_kw.columns = ['Kata Kunci', 'Jumlah']
                fig_bar = px.bar(
                    top_kw, x='Jumlah', y='Kata Kunci', orientation='h',
                    title="Top 10 Isu / Kata Kunci Terbanyak",
                    color='Jumlah', color_continuous_scale='Tealgrn'
                )
                fig_bar.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        # Grafik Tren Waktu (Jika kolom date tersedia)
        if 'date' in df_filtered.columns:
            df_time = df_filtered.copy()
            df_time['parsed_date'] = pd.to_datetime(df_time['date'], errors='coerce').dt.date
            daily_counts = df_time.dropna(subset=['parsed_date']).groupby('parsed_date').size().reset_index(name='Jumlah Berita')
            if len(daily_counts) > 1:
                fig_line = px.area(
                    daily_counts, x='parsed_date', y='Jumlah Berita',
                    title="Tren Volume Pemberitaan Harian",
                    color_discrete_sequence=['#38bdf8']
                )
                fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig_line, use_container_width=True)

        with st.expander("📋 Eksplorasi Data Mentah"):
            cols = [c for c in ['date', 'source', 'keyword', 'text', 'Kategori_Nama'] if c in df_filtered.columns]
            st.dataframe(df_filtered[cols].head(50), use_container_width=True)
    else:
        st.warning("Dataset CSV belum ditemukan di folder kerja Colab.")

# === TAB 3: ANALISIS MASSAL (UPLOAD CSV) ===
with menu_utama[2]:
    st.subheader("📂 Deteksi Massal dari File CSV")
    st.write("Unggah file CSV yang memiliki kolom judul berita (misalnya kolom text atau title) untuk diklasifikasikan sekaligus.")
    uploaded_csv = st.file_uploader("Pilih file CSV:", type=["csv"])

    if uploaded_csv is not None:
        df_up = pd.read_csv(uploaded_csv)
        st.write("Pratinjau Data:", df_up.head(5))
        target_col = st.selectbox("Pilih kolom yang berisi teks judul berita:", df_up.columns)

        if st.button("⚡ Proses Semua Baris", type="primary"):
            with st.spinner("Sedang menganalisis seluruh baris..."):
                cleaned_list = [clean_text(t) for t in df_up[target_col].astype(str)]
                seqs = tokenizer.texts_to_sequences(cleaned_list)
                pads = pad_sequences(seqs, maxlen=50, padding='post', truncating='post')
                preds = model.predict(pads, verbose=0)
                df_up['Prediksi_AI'] = [LABEL_MAP[int(np.argmax(p))] for p in preds]
                df_up['Keyakinan (%)'] = [round(float(np.max(p))*100, 2) for p in preds]

            st.success("Selesai! Berikut hasil klasifikasinya:")
            st.dataframe(df_up, use_container_width=True)
            csv_out = df_up.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Unduh Hasil Analisis (CSV)", data=csv_out, file_name="hasil_deteksi_siber.csv", mime="text/csv")
