# Acknowledgments

## AI Assistance
Analisis, debugging, dan penyusunan dokumentasi pada proyek ini dibantu oleh **DeepSeek V4.1 Flash Free** yang dijalankan melalui **OpenCode** sebagai asisten coding, mencakup:
- Penelusuran penyebab model kolaps pada iterasi pertama (`BatchNorm` momentum, `EarlyStopping` memantau metrik yang tepat)
- Penyusunan ulang sel training dan evaluasi (callback memantau `val_loss`, `ValDistributionLogger`)
- Penyusunan sel diagnostik (distribusi prediksi, statistik `BatchNorm`, mode train vs eval)
- Draf `README.md` dan rangkuman hasil

## Dataset
- **Dicoding Indonesia** — penyedia paket dataset `rockpaperscissors.zip` untuk proyek akhir *Belajar Machine Learning untuk Pemula* (versi ter-crop `rps-cv-images`, 2.188 gambar).  
  🔗 https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip
- **Julien de la Bruère-Terreault (`DrGFreeman`)** — kreator asli dataset Rock-Paper-Scissors (`rps-cv`).  
  🔗 https://github.com/DrGFreeman/rps-cv
- **Laurence Moroney** — kreator dataset *Rock Paper Scissors* publik lainnya (referensi tutorial TensorFlow).

## Tools & Libraries
- TensorFlow / Keras, NumPy, Matplotlib, scikit-learn, Pillow
- `uv` untuk manajemen dependensi dan virtual environment
- Git & GitHub untuk version control

---

**Seluruh keputusan akhir, eksekusi eksperimen, verifikasi hasil, serta tanggung jawab atas isi laporan tetap berada pada penulis.**