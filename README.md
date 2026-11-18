# model_summa

Server ekstraksi informasi struk untuk aplikasi Summa.
EasyOCR menghasilkan kotak dan teks, LayoutLMv3 memberi label, penengah
hibrida menggabungkannya, lalu keluarannya JSON sesuai `src/fitur/struk/tipe.ts`
di repo aplikasi.

## Endpoint

| Endpoint | Guna |
|---|---|
| `GET /` | Melaporkan `rss_gb`, `puncak_gb`, `detik_muat`, `berkunci` |
| `POST /ekstrak` | Kirim foto struk sebagai `berkas`, dapat JSON |

Balasan `/ekstrak` punya tiga bagian: `aplikasi` sesuai `tipe.ts`, `mentah`
persis keluaran `ke_json` Notebook E, dan `waktu` per tahap.

## Berkas

| Berkas | Asal | Boleh diubah? |
|---|---|---|
| `pembanding_aturan.py` | Kaggle, dibekukan | **Tidak** |
| `hibrida_penengah.py` | Kaggle, dibekukan | **Tidak** |
| `rantai.py` bagian atas | Salinan Notebook E sel 4, 7, 9 | Tidak, kecuali notebook ikut berubah |
| `rantai.py` bagian bawah | Penyesuai ke `tipe.ts`, bukan salinan | Boleh |
| `app.py` | Rantai Notebook E sel 8 ditambah penengah | Boleh |
| `uji.py` | Uji mandiri, jalan juga saat build | Boleh |

## Yang masih kurang

1. ~~Bobot model di Hugging Face~~ selesai.
   Sudah dipakai: `Klissh/layoutlmv3-struk-summa-ocr`, 43 label BIO.
2. ~~Versi `easyocr` dikunci~~ selesai: `easyocr==1.7.2`, sama dengan Notebook A dan E.
3. Latensi dan ketepatan rantai **dengan penengah** belum pernah diukur.
   Notebook E mengukur tanpa penengah.

## Batasan yang wajib ikut dilaporkan

Server memakai `penengah(..., satu_total=True)`, yaitu varian **P2U**.
Menurut catatan di `hibrida_penengah.py`, P2U dirancang setelah angka uji P2
terlihat, jadi angkanya pasca-hoc dan tidak boleh disajikan sebagai hasil uji
yang bersih. P2U dipakai karena aplikasi menuntut satu total per struk.

## Pengenal huruf

`PENGENAL` memilih siapa yang membaca teks di dalam kotak CRAFT. Koordinat
kotak tidak pernah berubah, jadi masukan ruang untuk LayoutLMv3 tetap sama.

| Nilai | Arti |
|---|---|
| `easyocr` | **bawaan.** Jalur lama, CRNN EasyOCR 1.7.2 |
| `ppocr` | kotak CRAFT dibaca ulang PP-OCRv4 lewat `rapidocr_onnxruntime` |

Plafon OCR pada 11 struk himpunan uji ketiga, yaitu berapa nilai jawaban benar
yang hadir utuh di dalam kotak:

| Pengenal | Plafon 66 medan | Waktu 11 struk |
|---|---|---|
| EasyOCR CRNN | 43 dari 66 = 65,2 persen | sekitar 65 dtk |
| PP-OCRv4 rec | **55 dari 66 = 83,3 persen** | 20 dtk |

Angka 43,9 persen di tag `diukur-uji3` diukur dengan `PENGENAL=easyocr`, dan
`ppocr` **belum pernah diukur pada data bersih**. Karena itu bawaannya tetap
`easyocr`. Salah setel menjatuhkan peladen saat hidup, bukan diam-diam
kembali ke EasyOCR.

## Keamanan

Kalau variabel `KUNCI` diisi di Railway, `/ekstrak` menolak permintaan yang
tidak membawa header `X-Kunci` yang sama. Kalau kosong, endpoint terbuka untuk
umum, dan siapa pun bisa menghabiskan kredit Anda.

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
