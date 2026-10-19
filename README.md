# model_summa

Server ekstraksi informasi struk untuk aplikasi Summa.
EasyOCR menghasilkan kotak dan teks, LayoutLMv3 memberi label, penengah
hibrida menggabungkannya, lalu keluarannya JSON sesuai `src/fitur/struk/tipe.ts`
di repo aplikasi.

## Keadaan sekarang: tahap 1

Baru memuat model dan melaporkan pemakaian memori. Endpoint ekstraksi belum ada.

| Endpoint | Guna |
|---|---|
| `GET /` | Melaporkan `rss_gb`, `puncak_gb`, dan `detik_muat` |

Tujuan tahap ini: mengetahui apakah model muat di paket Railway yang dipakai,
berapa lama mulai dingin, dan berapa besar image yang terbentuk.

## Yang masih kurang untuk tahap 2

1. `pembanding_aturan.py` dari Kaggle. Berkas ini dibekukan, jangan diubah.
2. `hibrida_penengah.py`. Wajib satu folder dengan berkas di atas.
3. Isi Notebook E sel 8, 9, dan 10: `urut_baca`, `norm_bbox`,
   `normalisasi_angka`, rantai ujung ke ujung, dan `ke_json`.
4. Bobot model di Hugging Face, lalu `MODEL_ID` diarahkan ke sana.

Rantai pra-pemrosesan harus disalin persis, satu huruf pun tidak boleh beda.
Kalau berbeda, keluaran server tidak lagi mewakili angka F1 di BAB IV.

## Pengaturan yang wajib dipertahankan

| Hal | Nilai | Alasan |
|---|---|---|
| `apply_ocr` pada processor | `False` | Kalau True, kotak EasyOCR ditimpa diam-diam |
| `width_ths` pada EasyOCR | `0.0` | Nilai bawaan 0,5 hanya menghasilkan 53,5 kotak per struk, bukan 97,4 |
| Urutan `id2label` | Sama dengan skema | Kalau tidak, seluruh label salah tanpa pesan error |
| Urutan kotak | Model memakai `urut_baca()`, aturan memakai urutan asli | Kalau tertukar, penengah salah sasaran |

## Jalankan di komputer sendiri

```bash
docker build -t model-summa .
docker run -p 8000:8000 -e PORT=8000 model-summa
curl http://localhost:8000/
```
