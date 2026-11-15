# 🛡️ Wasabil Corefine AI
**Cyber Cognitive Defense System: Threat & Disinformation Intelligence**

[![Streamlit App](https://streamlit.io)](https://streamlit.app)


**Wasabil Corefine AI** adalah sistem pemantauan isu siber *real-time* dan klasifikasi otomatis berbasis *Deep Learning* (**Bidirectional LSTM v2.2**) yang dikembangkan oleh **Tim Wasabil Codeone** untuk **Kartika Cyber Hackathon 2026**. 

Sistem ini dirancang untuk mendeteksi secara dini potensi konflik sosial, disinformasi, propaganda, dan ancaman keamanan siber yang beredar di portal berita daring Indonesia.

---

## 🌐 Akses Live Aplikasi
* **Live Dashboard (Streamlit Cloud):** https://streamlit.app


---

## ✨ Fitur Utama
1. **🔍 Uji Deteksi Real-Time**
   * **Input Teks Judul/Klaim:** Menganalisis narasi berita secara instan dan memberikan skor keyakinan AI (*confidence gauge*) serta distribusi probabilitas 3 kelas (*Aman/Netral*, *Potensi Konflik*, dan *Propaganda/Disinformasi*).
   * **Auto-Scraping Tautan (URL) Berita:** Mengekstrak judul dan isi artikel langsung dari tautan berita daring untuk didiagnosis secara otomatis.
2. **📊 Dashboard Intelijen Data**
   * Visualisasi interaktif terhadap **9.941 artikel berita** yang dikumpulkan dari **1.555 sumber media** berdasarkan **34 kata kunci ancaman**.
   * Dilengkapi filter topik kata kunci, proporsi tingkat ancaman, tren publikasi berdasarkan waktu, 10 kata kunci terbanyak, dan eksplorasi tabel data mentah.
3. **📁 Analisis Massal (CSV)**
   * Memungkinkan pengguna mengunggah berkas CSV berisi ratusan hingga ribuan judul berita untuk diklasifikasikan secara bersamaan dalam satu kali proses.

---

## 🏗️ Arsitektur & Alur Kerja Sistem

```text
[1.1_scraping_cyber_alert.ipynb] ──► Mengumpulkan 9.941 Artikel Berita (dataset_cyber_alerts_LATEST.csv)
                                                    │
                                                    ▼
[2.2_cyber_load.ipynb]           ──► Preprocessing NLP & Training Model BiLSTM v2.2 + Baseline LogReg
                                                    │
                                                    ▼
[app.py (Streamlit Cloud)]       ──► Inferensi Real-Time, URL Scraper & Dashboard Intelijen Siber
```

---

## 📂 Struktur Repositori
* `app.py` — Kode utama antarmuka web dan inferensi model yang di-deploy ke Streamlit Cloud.
* `1.1_scraping_cyber_alert.ipynb` — Modul *automated data pipeline* untuk pengumpulan dan ekstraksi berita daring berdasarkan rentang tanggal dan kata kunci ancaman.
* `2.2_cyber_load.ipynb` — Modul *preprocessing* teks, pelatihan model *Deep Learning* (BiLSTM), dan evaluasi model.
* `model_cyber_bilstm.keras` — Bobot dan arsitektur model utama *Bidirectional LSTM v2.2*.
* `tokenizer_cyber.json` & `model_config.json` — Konfigurasi *tokenizer* dan parameter pemrosesan teks.
* `model_cyber_logreg.joblib` & `tfidf_vectorizer.joblib` — Model pembanding (*Logistic Regression* + *TF-IDF*).
* `dataset_cyber_alerts_LATEST.csv` — Dataset hasil *crawling* sebanyak 9.941 artikel berita.
* `requirements.txt` — Daftar dependensi *library* Python untuk lingkungan produksi.

---

## 🛠️ Teknologi yang Digunakan
* **Bahasa Pemrograman:** Python 3
* **Deep Learning & NLP:** TensorFlow / Keras (Bidirectional LSTM), Scikit-Learn (TF-IDF & Logistic Regression)
* **Data Scraping & Processing:** BeautifulSoup4, Requests, Pandas, NumPy
* **Visualisasi & Web App:** Streamlit, Plotly

---

## 👥 Tim Pengembang
**Wasabil Codeone** — *Kartika Cyber Hackathon 2026*
