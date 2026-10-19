# -*- coding: utf-8 -*-
"""Penengah hibrida: model LayoutLMv3 + pembanding aturan.

Dipilih di data dev sesuai claude/pra-registrasi-hibrida.md (kebijakan P2).
Varian P2U menambahkan kendala satu total per struk; lihat catatan di bawah.

Pemakaian:
    from hibrida_penengah import penengah
    label_akhir = penengah(struk, label_model, skor_model)

    struk        dict dengan kunci "kotak" (daftar {teks, bbox}), "tinggi", "lebar"
    label_model  daftar label datar dari model, urutannya = struk["kotak"]
    skor_model   opsional, daftar dict {label_datar: logit} per kotak.
                 Hanya dipakai oleh kendala keunikan untuk memilih kandidat
                 terbaik. Boleh None.
"""
import pembanding_aturan as PA

# Label yang presisi aturannya di DEV mencapai 1,000. Ditetapkan di data dev,
# tidak pernah disetel memakai data uji.
SET_ATURAN_MENANG = {
    "sub_total.discount_price",
    "sub_total.service_price",
    "sub_total.subtotal_price",
    "total.cashprice",
    "total.menuqty_cnt",
    "total.total_price",
}

def _satu_total(kotak, label, aturan, skor):
    """Kendala domain: satu struk tepat satu total.total_price."""
    label = list(label)
    punya = [i for i, L in enumerate(label) if L == "total.total_price"]
    if len(punya) == 1:
        return label
    r_tot = [i for i, L in enumerate(aturan) if L == "total.total_price"]
    if r_tot:
        pilih = r_tot[0]
    elif len(punya) > 1 and skor:
        pilih = max(punya, key=lambda i: skor[i].get("total.total_price", -1e9))
    elif len(punya) > 1:
        pilih = max(punya, key=lambda i: kotak[i]["bbox"][1])   # paling bawah
    else:
        return label
    for i in punya:
        if i == pilih: continue
        if skor:
            alt = {k: v for k, v in skor[i].items() if k != "total.total_price"}
            label[i] = max(alt, key=alt.get) if alt else "O"
        else:
            label[i] = "O"
    label[pilih] = "total.total_price"
    return label

def penengah(struk, label_model, skor_model=None, satu_total=True):
    """Kembalikan label akhir, urutannya sama dengan struk["kotak"].

    satu_total=True memakai varian P2U. Lihat BATASAN di bawah.
    """
    aturan = PA.prediksi(struk)
    assert len(aturan) == len(label_model) == len(struk["kotak"])
    out = [aturan[i] if aturan[i] in SET_ATURAN_MENANG else label_model[i]
           for i in range(len(label_model))]
    if satu_total:
        out = _satu_total(struk["kotak"], out, aturan, skor_model)
    return out

# ---------------------------------------------------------------------------
# BATASAN YANG WAJIB IKUT DILAPORKAN
#
# P2  (satu_total=False) adalah kebijakan yang dipilih lebih dulu di data dev
#     lalu diuji satu kali. Angkanya sah sebagai hasil uji.
#
# P2U (satu_total=True) menambahkan kendala keunikan. Gabungan ini TIDAK
#     termasuk lima kebijakan yang dipra-registrasikan, dan dirancang setelah
#     angka uji P2 terlihat. Angka ujinya PASCA-HOC dan tidak boleh disajikan
#     sebagai hasil uji yang bersih.
#
#     Dipakai di aplikasi karena aplikasi menuntut satu jawaban per struk,
#     bukan karena angkanya lebih bagus. Pada tingkat kotak P2U justru sedikit
#     lebih rendah daripada P2.
# ---------------------------------------------------------------------------
