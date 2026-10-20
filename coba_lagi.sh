#!/bin/sh
# Tunggu Railway selesai membangun ulang, lalu kirim struk yang sama.
K=tXZrs9V6yzZ2U2KS8KJH3CUyyMsDWUdi
A=https://modelsumma-production.up.railway.app
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  sleep 30
  curl -s --max-time 120 -X POST -H "X-Kunci: $K" -F "berkas=@struk_uji.png" "$A/ekstrak" -o hasil_baru.json
  if python -c "import json,sys; d=json.load(open('hasil_baru.json')); sys.exit(0 if 'mentah' in d else 1)" 2>/dev/null; then
    echo "=== percobaan ke-$i, $(( i * 30 )) detik setelah push ==="
    python -c "
import json
d = json.load(open('hasil_baru.json'))['mentah']
print('toko:', d['toko'])
print('item:', len(d['item']))
for it in d['item']:
    print(f\"  {str(it['nama'])[:38]:38} harga={it['harga']}\")
for m in ['subtotal','pajak','total','tunai','kembalian']:
    v = d[m]
    print(f\"{m:10} {v['nilai'] if v else None}\")
"
    exit 0
  fi
done
echo "server tidak menjawab dalam 6 menit"
