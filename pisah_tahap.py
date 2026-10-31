"""OCR dua tahap, untuk struk yang barisnya terlalu rapat bagi CRAFT.

Latar belakangnya terukur. Pada struk Lotus's 31 Okt 2026, jarak antarbaris
25 piksel sedangkan tinggi hurufnya 24,5 piksel, jadi kertas kosong yang
tersisa antarbaris cuma sekitar setengah piksel. Pada struk 99 Speed Mart
jaraknya 34 piksel dengan tinggi huruf 23 piksel, yaitu sela sekitar 11
piksel. CRAFT menyatukan baris yang selanya setengah piksel, lalu memotongnya
menegak, sehingga kolom harga tidak pernah terbentuk dan nol harga item
terbaca.

Pelurusan lengkungan sudah dicoba dan TIDAK menolong: nisbah lembah profil
Lotus's 0,140 kalau hanya diputar dan 0,136 kalau diluruskan penuh, yaitu
selisih sebesar derau, sedangkan harga benar justru turun dari 20 jadi 16 dari
21 karena interpolasinya mengaburkan angka. Jadi di sini gambarnya hanya
diputar.

Yang menolong adalah memberi gambar BERBEDA kepada dua tahap OCR:

  - deteksi kotak pada gambar yang diregangkan tegak dua kali, supaya selanya
    melebar jadi sekitar satu piksel dan CRAFT berhenti menyatukan baris
  - pengenalan huruf pada gambar yang TIDAK diregangkan, sebab huruf yang
    teregang tidak lagi seperti huruf yang pernah dilatihkan ke pengenalnya

Diukur pada struk Lotus's: 0 dari 21 harga jadi 16 dari 21, dan daftar
barangnya dari 5 nama sampah jadi 19 nama terbaca. Tambahan waktunya sekitar
1,4 detik di laptop.

Gambar yang diputar ikut dikembalikan, dan gambar itu pula yang harus
diumpankan ke LayoutLMv3, supaya kotak dan gambar memakai satu tata koordinat.
"""
import cv2
import numpy as np
from PIL import Image

import rantai

# shortcut: regangan tetap 2x, cukup untuk sela setengah piksel yang terukur
# di struk Lotus's; kalau nanti ada struk yang lebih rapat lagi, turunkan
# nilainya dari jarak baris yang terukur, bukan dari tetapan.
FY = 2
SUDUT_MAKS, SUDUT_LANGKAH = 6.0, 0.25
SISA_PITA = 0.005


def _peta_teks(abu, jendela=9, persentil=92):
    """Piksel yang ragam lokalnya tinggi, yaitu goresan huruf.

    Ambang gelap terang tidak bisa dipakai: pada foto struk di atas lantai
    ubin, Otsu menyatukan kertas dengan lantai dan tangan ikut terhitung
    tinta. Ragam lokal hanya menyala di tempat ada goresan.
    """
    f = abu.astype(np.float32)
    rata = cv2.boxFilter(f, -1, (jendela, jendela))
    rata2 = cv2.boxFilter(f * f, -1, (jendela, jendela))
    sd = np.sqrt(np.maximum(rata2 - rata * rata, 0.0))
    return (sd > np.percentile(sd, persentil)).astype(np.uint8)


def _pita_teks(m, sisa=SISA_PITA):
    """Rentang x badan teks, dari kuantil massa tinta per kolom.

    Kuantil, bukan ambang: latar bertekstur menyumbang tinta tipis di
    mana-mana, dan ambang apa pun akan ikut menangkapnya.
    """
    kolom = m.sum(axis=0).astype(np.float64)
    if kolom.sum() <= 0:
        return 0, m.shape[1]
    kum = np.cumsum(kolom) / kolom.sum()
    x0 = int(np.searchsorted(kum, sisa))
    x1 = int(np.searchsorted(kum, 1.0 - sisa)) + 1
    return x0, max(x0 + 8, x1)


def _sudut(m, x0, x1):
    """Sudut yang membuat profil baris paling bergerigi, yaitu paling lurus.

    Kalau barisnya mendatar, tiap baris menyumbang satu puncak tajam dan
    ragam profilnya besar. Kalau miring, puncaknya saling tumpah dan
    ragamnya kecil.
    """
    H, W = m.shape
    terbaik, sudut = -1.0, 0.0
    for a in np.arange(-SUDUT_MAKS, SUDUT_MAKS + 1e-9, SUDUT_LANGKAH):
        r = cv2.warpAffine(m, cv2.getRotationMatrix2D((W / 2, H / 2), float(a), 1.0), (W, H))
        v = float(np.var(r[:, x0:x1].sum(axis=1).astype(np.float32)))
        if v > terbaik:
            terbaik, sudut = v, float(a)
    return sudut


def n_harga_item(hasil):
    """Berapa item yang dapat harga. Nol berarti daftar barangnya tidak berguna."""
    return sum(1 for x in (hasil.get("item") or [])
               if isinstance(x.get("harga"), (int, float)) and not isinstance(x.get("harga"), bool))


def perlu(hasil):
    """Pemicu. Sengaja sesempit mungkin: hanya saat TIDAK SATU PUN item berharga.

    Pemicu dari kotak OCR saja sudah dicoba dan ditolak, sebab tidak ada
    ukuran yang memisahkan dengan bersih: kotak mirip uang di 45 persen kanan
    gambar 11,1 persen pada Lotus's lawan 18,6 dan 19,1 persen pada dua struk
    Speed Mart, yaitu gradasi, bukan jurang. Ambang di situ akan menyala pada
    struk yang sudah benar.

    Atas 43 struk uji, pemicu ini menyala pada 4 struk, dan keempatnya memang
    sudah nol harga item sebelumnya.
    """
    return hasil is not None and n_harga_item(hasil) == 0


def ocr(jalur, pembaca):
    """(gambar PIL yang sudah diputar, daftar kotak). (None, []) kalau gagal.

    Bentuk kotaknya sama dengan ocr_mentah di app.py, supaya sisa rantainya
    tidak perlu tahu kotak itu datang dari jalur mana.
    """
    abu = cv2.imread(jalur, cv2.IMREAD_GRAYSCALE)
    if abu is None or abu.size == 0:
        return None, []
    m = _peta_teks(abu)
    x0, x1 = _pita_teks(m)
    H, W = abu.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), _sudut(m, x0, x1), 1.0)
    diputar = cv2.warpAffine(abu, M, (W, H), flags=cv2.INTER_CUBIC,
                             borderValue=int(np.median(abu)))

    teregang = cv2.resize(diputar, None, fx=1, fy=FY, interpolation=cv2.INTER_CUBIC)
    datar, _ = pembaca.detect(cv2.cvtColor(teregang, cv2.COLOR_GRAY2RGB),
                              width_ths=0.0, link_threshold=0.4)
    if not datar or not datar[0]:
        return None, []
    # Skala y dikembalikan, sebab pengenalan dikerjakan di gambar yang tidak
    # teregang. detect memberi [x_kiri, x_kanan, y_atas, y_bawah].
    kk = [[int(xa), int(xb), int(ya / FY), int(yb / FY)] for xa, xb, ya, yb in datar[0]]

    kotak = []
    for poly, teks, skor in pembaca.recognize(diputar, horizontal_list=kk, free_list=[],
                                              detail=1, paragraph=False):
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        kotak.append(dict(
            bbox=[int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
            teks=rantai.normalisasi_angka(teks), teks_asli=teks, skor=float(skor)))
    kotak.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    return Image.fromarray(cv2.cvtColor(diputar, cv2.COLOR_GRAY2RGB)), kotak