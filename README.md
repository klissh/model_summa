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
| `ppocr` | **bawaan sejak 18 November 2026.** Deteksi CRAFT saja, lalu dibaca PP-OCRv4 lewat `rapidocr_onnxruntime` |
| `easyocr` | jalur lama, CRNN EasyOCR 1.7.2 |

`UTAS_PENGENAL` mengatur utas onnxruntime, bawaannya 4.

Diukur pada rantai PENUH di laptop, yaitu EasyOCR lalu LayoutLMv3 lalu
`rantai.py`, 11 struk himpunan uji ketiga, **dua lari tiap pengenal**:

| Pengenal | Ketepatan 8 medan | Waktu per struk |
|---|---|---|
| EasyOCR CRNN | 35 dari 66 = 53,0 persen, kedua lari sama | 11,6 dan 13,3 dtk |
| **PP-OCRv4 rec** | **39 dari 66 = 59,1 persen**, kedua lari sama | 11,0 dan 13,1 dtk |

Ketepatannya **stabil** antarlari, sedangkan **sebaran latensinya bertumpang
tindih**, jadi yang bisa diklaim: `ppocr` memberi +4 medan **tanpa menambah
latensi yang terukur**. Ia tidak boleh disebut lebih cepat.

Latensi itu baru setara setelah dua pemangkasan. Sebelumnya `ppocr` 21,6 detik
per struk:

1. `kotak_craft` memakai `pembaca.detect()` saja, jadi pengenal CRNN EasyOCR
   tidak dijalankan sama sekali, sebab teksnya akan ditimpa seluruhnya.
   Kotaknya terukur sama dengan `readtext` kecuali 3 kotak dari 1326 yang
   bergeser 1 piksel, selisih yang `readtext` sendiri pun tidak mengulang
   persis antarjalan. Hemat 1,4 sampai 1,8 detik.
2. `RapidOCR` diberi `intra_op_num_threads` eksplisit. Inilah sebagian besar
   penghematannya, dan besarnya melampaui perkiraan.

Plafon OCR, yaitu berapa nilai jawaban benar yang hadir utuh di dalam kotak:

| Pengenal | Plafon 66 medan |
|---|---|
| EasyOCR CRNN | 43 dari 66 = 65,2 persen |
| PP-OCRv4 rec | **55 dari 66 = 83,3 persen** |

### Yang perlu diketahui tentang bawaan ini

Huruf di luar ASCII dibuang dari keluaran pengenal. Model PP-OCRv4 bawaan
`rapidocr` adalah model **Tionghoa ditambah Inggris**, dan pada potongan yang
berderau ia mengeluarkan huruf Han: terukur 17 dari 1326 kotak uji3 memuat
huruf non-ASCII dan 12 di antaranya huruf Han, sedangkan EasyOCR nol. Struk
Malaysia tidak memuat huruf Han, dan LayoutLMv3 dilatih pada teks beraksara
Latin.

Kalau `rapidocr_onnxruntime` tidak terpasang, perlakuannya berbeda menurut
siapa yang memilih:

| Keadaan | Perlakuan |
|---|---|
| `PENGENAL=ppocr` disetel eksplisit | **gagal keras**, peladen menolak hidup |
| `ppocr` dipakai dari bawaan | **mundur** ke EasyOCR dengan peringatan, dan `sehat` menyebutkan kemundurannya |

Angka 43,9 persen di tag `diukur-uji3` diukur dengan `PENGENAL=easyocr`, dan
`ppocr` **belum pernah diukur pada data bersih** maupun berjalan di peladen
sungguhan. Untuk mengulang pengukuran tag itu, setel `PENGENAL=easyocr`.

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
