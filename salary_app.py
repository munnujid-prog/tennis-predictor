"""
Peramal Gaji Persaraan — Gred DH12 hingga DH14
Jabatan Pendidikan Politeknik dan Kolej Komuniti (KPPK)
Berasaskan Pekeliling Perkhidmatan & Jadual Gaji Minimum-Maksimum
Jalankan dengan:  streamlit run salary_app.py
"""

import datetime
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Peramal Gaji Persaraan DH KPPK",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

* { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #0a0e27 0%, #0d1b4b 50%, #0a2a4a 100%);
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b4b 0%, #1a2f6e 100%);
    border-right: 2px solid #3b82f6;
}
[data-testid="stSidebar"] * { color: #bfdbfe !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #93c5fd !important; }
[data-testid="stSidebar"] hr { border-color: #3b82f6 !important; }

/* Hero */
.hero {
    background: linear-gradient(90deg, #1d4ed8 0%, #2563eb 40%, #1e40af 100%);
    border-radius: 20px; padding: 32px 40px; margin-bottom: 28px;
    text-align: center; box-shadow: 0 8px 40px rgba(37,99,235,0.5);
    border: 1px solid rgba(147,197,253,0.2);
}
.hero h1 { color: #fff !important; font-size: 2.2rem !important; font-weight: 900 !important; margin: 0; letter-spacing: 0.5px; }
.hero p  { color: rgba(255,255,255,0.88) !important; font-size: 1rem; margin: 8px 0 0 0; }
.hero .sub { color: #93c5fd !important; font-size: 0.88rem; margin-top: 6px; }

/* Section title */
.sec-title {
    font-size: 1rem; font-weight: 700; color: #93c5fd;
    border-left: 4px solid #3b82f6; padding-left: 10px;
    margin: 22px 0 14px 0; text-transform: uppercase; letter-spacing: 1px;
}

/* Card */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(147,197,253,0.2);
    border-radius: 16px; padding: 20px 22px; margin-bottom: 14px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.card h4 { color: #93c5fd !important; margin: 0 0 10px 0; font-size: 0.9rem;
            text-transform: uppercase; letter-spacing: 1.5px; }

/* Result big number */
.result-main {
    background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 100%);
    border: 2px solid #3b82f6; border-radius: 18px;
    padding: 24px 28px; text-align: center; margin-bottom: 16px;
    box-shadow: 0 6px 30px rgba(37,99,235,0.4);
}
.result-main .label { color: #93c5fd; font-size: 0.82rem; letter-spacing: 2px;
                       text-transform: uppercase; margin-bottom: 4px; }
.result-main .amount { color: #fff; font-size: 2.8rem; font-weight: 900; line-height: 1; }
.result-main .sub-amount { color: #bfdbfe; font-size: 1rem; margin-top: 6px; }

/* Metric chip */
.chip-row { display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0; }
.chip {
    background: rgba(59,130,246,0.15); border: 1px solid rgba(147,197,253,0.3);
    border-radius: 20px; padding: 5px 14px; font-size: 0.82rem; color: #bfdbfe;
}
.chip-gold {
    background: rgba(251,191,36,0.15); border: 1px solid rgba(251,191,36,0.4);
    border-radius: 20px; padding: 5px 14px; font-size: 0.82rem; color: #fde68a;
}
.chip-green {
    background: rgba(16,185,129,0.15); border: 1px solid rgba(52,211,153,0.4);
    border-radius: 20px; padding: 5px 14px; font-size: 0.82rem; color: #6ee7b7;
}

/* Grade badge */
.grade-badge {
    display: inline-block; border-radius: 12px; padding: 6px 18px;
    font-weight: 900; font-size: 1rem; margin: 4px;
}

/* Increment table */
.inc-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
.inc-num { color: #fde68a; font-weight: 700; min-width: 28px; }
.inc-bar-wrap { flex: 1; background: rgba(255,255,255,0.07); border-radius: 6px; height: 10px; }
.inc-bar { height: 10px; border-radius: 6px; }
.inc-val { color: #6ee7b7; font-size: 0.85rem; font-weight: 600; min-width: 90px; text-align: right; }

/* Divider */
.divider {
    height: 2px;
    background: linear-gradient(90deg,#1d4ed8,#3b82f6,#93c5fd,#3b82f6,#1d4ed8);
    border-radius: 2px; margin: 22px 0;
}

/* Disclaimer */
.disclaimer {
    background: rgba(251,191,36,0.08); border: 1px solid rgba(251,191,36,0.3);
    border-radius: 12px; padding: 14px 18px; color: #fde68a; font-size: 0.85rem;
}

/* Widget labels */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label,
[data-testid="stRadio"] label { color: #93c5fd !important; font-weight: 600 !important; }

/* Dropdown popup */
[data-baseweb="popover"], [data-baseweb="popover"] * { background:#fff !important; color:#111 !important; }
[data-baseweb="menu"],    [data-baseweb="menu"] *    { background:#fff !important; color:#111 !important; }
[role="option"], [role="option"] * { background:#fff !important; color:#111 !important; }
[role="option"]:hover, [role="option"]:hover * { background:#eff6ff !important; color:#000 !important; }
div[data-baseweb="select"] > div { background:rgba(255,255,255,0.07) !important; border:1px solid rgba(147,197,253,0.4) !important; color:#fff !important; border-radius:8px !important; }
div[data-baseweb="select"] span { color:#fff !important; }

/* Button */
[data-testid="stButton"] > button {
    background: linear-gradient(90deg, #1d4ed8, #3b82f6) !important;
    color: #fff !important; font-size: 1.05rem !important; font-weight: 700 !important;
    border: none !important; border-radius: 10px !important;
    padding: 12px 36px !important; width: 100% !important;
    box-shadow: 0 4px 16px rgba(37,99,235,0.45) !important;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(90deg, #1e40af, #2563eb) !important;
    box-shadow: 0 6px 24px rgba(37,99,235,0.6) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Salary Data: Pensyarah Politeknik & Kolej Komuniti ───────────────────────
# Skim Perkhidmatan Pensyarah — Kumpulan Pengurusan & Profesional (P&P)
# Jadual Gaji SSM 2024 — Pekeliling Perkhidmatan Bil. 1/2024
# Gaji semasa maksimum ditetapkan RM 30,000 (gaji tertinggi dalam tangga gred)

GRED_DATA = {
    "DH12": {
        "nama": "Pensyarah (Gred DH12)",
        "jawatan": "Pensyarah — Kolej Komuniti (Lepasan Diploma)",
        "gaji_min": 2422.28,
        "gaji_max": 30_000.00,
        "increment_base": 175.00,
        "anjakan_count": 3,
        "anjakan_amount": 300.00,
        "color": "#3b82f6",
        "bg": "rgba(59,130,246,0.15)",
        "syarat_persaraan": "Genap 55 tahun ATAU cukup 30 tahun perkhidmatan",
        "nota": "Boleh dipanjangkan atas pilihan sehingga umur 60 tahun",
        "kenaikan_pangkat": "Layak ke DH14 selepas 3 tahun dengan LP/Cemerlang",
    },
    "DH14": {
        "nama": "Pensyarah (Gred DH14)",
        "jawatan": "Pensyarah — Politeknik / Kolej Komuniti (Lepasan Ijazah)",
        "gaji_min": 3168.93,
        "gaji_max": 30_000.00,
        "increment_base": 200.00,
        "anjakan_count": 3,
        "anjakan_amount": 375.00,
        "color": "#8b5cf6",
        "bg": "rgba(139,92,246,0.15)",
        "syarat_persaraan": "Genap 55 tahun ATAU cukup 30 tahun perkhidmatan",
        "nota": "Boleh dipanjangkan atas pilihan sehingga umur 60 tahun",
        "kenaikan_pangkat": "Layak ke DH22 selepas 4 tahun dengan LP/Cemerlang",
    },
    "DH22": {
        "nama": "Pensyarah Kanan (Gred DH22)",
        "jawatan": "Pensyarah Kanan — Politeknik (Kepakaran / Pengalaman)",
        "gaji_min": 5_183.74,
        "gaji_max": 30_000.00,
        "increment_base": 275.00,
        "anjakan_count": 3,
        "anjakan_amount": 500.00,
        "color": "#10b981",
        "bg": "rgba(16,185,129,0.15)",
        "syarat_persaraan": "Genap 55 tahun ATAU cukup 30 tahun perkhidmatan",
        "nota": "Boleh dipanjangkan atas pilihan sehingga umur 60 tahun",
        "kenaikan_pangkat": "Layak ke DH26 selepas 4 tahun dengan rekod cemerlang",
    },
    "DH26": {
        "nama": "Pensyarah Utama (Gred DH26)",
        "jawatan": "Pensyarah Utama / Ketua Program — Politeknik",
        "gaji_min": 7_427.34,
        "gaji_max": 30_000.00,
        "increment_base": 350.00,
        "anjakan_count": 3,
        "anjakan_amount": 650.00,
        "color": "#f59e0b",
        "bg": "rgba(245,158,11,0.15)",
        "syarat_persaraan": "Genap 55 tahun ATAU cukup 30 tahun perkhidmatan",
        "nota": "Boleh dipanjangkan atas pilihan sehingga umur 60 tahun",
        "kenaikan_pangkat": "Gred tertinggi skim pensyarah — kenaikan ke jawatan pentadbiran (N/F)",
    },
}

# Kadar pencen berdasarkan tempoh perkhidmatan (Akta Pencen 1980)
def kadar_pencen(tahun_khidmat):
    if tahun_khidmat < 10:
        return 0.0
    elif tahun_khidmat < 25:
        return min(tahun_khidmat * 0.013333, 0.35)
    else:
        return min(0.35 + (tahun_khidmat - 25) * 0.02, 0.60)

def format_rm(nilai):
    return f"RM {nilai:,.2f}"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Peramal Gaji Persaraan")
    st.markdown("---")
    st.markdown("**Skop aplikasi:**")
    st.markdown("- Jabatan Pendidikan Politeknik dan Kolej Komuniti (KPPK)")
    st.markdown("- **Pensyarah** Gred DH12, DH14, DH22, DH26")
    st.markdown("- Persaraan umur 55–60 tahun")
    st.markdown("- Gaji semasa: RM 2,422 – RM 30,000")
    st.markdown("---")
    st.markdown("**Rujukan:**")
    st.markdown("- Pekeliling Perkhidmatan Bil. 1/2024 (SSM 2024)")
    st.markdown("- Akta Pencen 1980")
    st.markdown("- Skim Perkhidmatan DH (KPPK)")
    st.markdown("---")
    st.markdown("**Kadar pencen max:** 60% daripada gaji akhir")
    st.markdown("**Tempoh minimum:** 10 tahun perkhidmatan")
    st.markdown("---")
    st.caption("⚠️ Untuk anggaran sahaja. Sila rujuk JPA / Pejabat Kewangan untuk pengesahan rasmi.")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎓 Peramal Gaji Persaraan Pensyarah</h1>
  <p>Jabatan Pendidikan Politeknik dan Kolej Komuniti (KPPK) &nbsp;|&nbsp; Gred DH12 – DH26</p>
  <div class="sub">Gaji Semasa: RM 2,422 – RM 30,000 &nbsp;·&nbsp; SSM 2024 &nbsp;·&nbsp; Akta Pencen 1980</div>
</div>
""", unsafe_allow_html=True)
# ── Input Section ─────────────────────────────────────────────────────────────
HARI_INI = datetime.date.today()


st.markdown('<div class="sec-title">📋 Maklumat Peribadi & Perkhidmatan</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    nama = st.text_input("Nama Penuh", placeholder="Cth: Siti Aishah binti Ahmad")
with col2:
    gred_pilih = st.selectbox("Gred Jawatan Semasa", list(GRED_DATA.keys()),
                               format_func=lambda g: f"{g} — {GRED_DATA[g]['nama']}")
with col3:
    tarikh_lahir = st.date_input(
        "Tarikh Lahir",
        value=datetime.date(1979, 1, 1),
        min_value=datetime.date(1955, 1, 1),
        max_value=datetime.date(HARI_INI.year - 25, HARI_INI.month, HARI_INI.day),
        format="DD/MM/YYYY",
    )
    # Kira umur tepat berdasarkan tarikh lahir
    umur_sekarang = (HARI_INI - tarikh_lahir).days // 365
    st.markdown(
        f'<div style="color:#6ee7b7;font-size:0.82rem;margin-top:4px;">'
        f'📅 Umur semasa: <b>{umur_sekarang} tahun</b> '
        f'({tarikh_lahir.strftime("%d %b %Y")})</div>',
        unsafe_allow_html=True
    )

col4, col5, col6 = st.columns(3)
with col4:
    gaji_sekarang = st.number_input("Gaji Semasa (RM)",
                                     min_value=float(GRED_DATA[gred_pilih]["gaji_min"]),
                                     max_value=float(GRED_DATA[gred_pilih]["gaji_max"]),
                                     value=float(GRED_DATA[gred_pilih]["gaji_min"] + 400),
                                     step=50.0,
                                     format="%.2f")
with col5:
    tarikh_mula = st.date_input(
        "Tarikh Mula Berkhidmat",
        value=datetime.date(2000, 1, 1),
        min_value=datetime.date(1980, 1, 1),
        max_value=HARI_INI,
        format="DD/MM/YYYY",
    )
    tahun_mula = tarikh_mula.year
    tempoh_khidmat_kini = (HARI_INI - tarikh_mula).days // 365
    st.markdown(
        f'<div style="color:#93c5fd;font-size:0.82rem;margin-top:4px;">'
        f'⏳ Tempoh berkhidmat: <b>{tempoh_khidmat_kini} tahun</b></div>',
        unsafe_allow_html=True
    )
with col6:
    umur_persaraan = st.selectbox("Umur Persaraan Dirancang", [55, 56, 57, 58, 59, 60], index=0)
    # Kira tarikh persaraan anggaran
    tahun_persaraan_est = tarikh_lahir.year + umur_persaraan
    bulan_persaraan_est = tarikh_lahir.month
    st.markdown(
        f'<div style="color:#fde68a;font-size:0.82rem;margin-top:4px;">'
        f'🎓 Anggaran bersara: <b>{bulan_persaraan_est:02d}/{tahun_persaraan_est}</b></div>',
        unsafe_allow_html=True
    )

col7, col8 = st.columns(2)
with col7:
    pencen_pilihan = st.selectbox("Jenis Faedah Persaraan",
                                   ["Pencen Bulanan (Faedah Pencen)", "Wang Tunai (Gantian Cuti Rehat)", "Kedua-dua sekali"])
with col8:
    kenaikan_gaji_tahunan = st.selectbox("Kadar Kenaikan Gaji Tahunan (IKT)",
                                          ["Cemerlang (2x setahun)", "Baik (1x setahun)", "Memuaskan (1x setiap 2 tahun)"],
                                          index=1)

# ── Compute Button ────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
kira = st.button("🔍 KIRA GAJI PERSARAAN SAYA")

if kira:
    gred = GRED_DATA[gred_pilih]
    tahun_sekarang = 2024
    tahun_persaraan = tahun_sekarang + (umur_persaraan - umur_sekarang)
    tahun_khidmat_sekarang = tahun_sekarang - tahun_mula
    tahun_khidmat_persaraan = tahun_persaraan - tahun_mula

    # Kira increment setahun
    increment_map = {
        "Cemerlang (2x setahun)": gred["increment_base"] * 2,
        "Baik (1x setahun)": gred["increment_base"],
        "Memuaskan (1x setiap 2 tahun)": gred["increment_base"] / 2,
    }
    increment_tahunan = increment_map[kenaikan_gaji_tahunan]

    tahun_tinggal = umur_persaraan - umur_sekarang

    # Kira gaji akhir (progresif dengan ceiling)
    gaji = gaji_sekarang
    gaji_tahunan = []
    for t in range(tahun_tinggal):
        gaji = min(gaji + increment_tahunan, gred["gaji_max"])
        gaji_tahunan.append(gaji)

    gaji_akhir = gaji_tahunan[-1] if gaji_tahunan else gaji_sekarang

    # Kadar pencen
    kadar = kadar_pencen(tahun_khidmat_persaraan)
    pencen_bulanan = gaji_akhir * kadar

    # Gantian Cuti Rehat (GCR) — max 90 hari
    gcr = gaji_akhir / 26 * 90  # 90 hari maksimum

    # Derivasi bayaran ex-gratia (anggaran)
    ex_gratia = gaji_akhir * 3  # anggaran kasar

    # ── Grade badge row ───────────────────────────────────────────────────────
    badge_html = ""
    for g, d in GRED_DATA.items():
        is_active = g == gred_pilih
        style = f"background:{d['color']};color:#fff;" if is_active else f"background:{d['bg']};color:{d['color']};border:1px solid {d['color']};"
        badge_html += f'<span class="grade-badge" style="{style}">{g}</span>'

    st.markdown(f'<div style="margin-bottom:16px;">{badge_html}</div>', unsafe_allow_html=True)

    # ── Nama & gred info ──────────────────────────────────────────────────────
    nama_papar = nama.strip() if nama.strip() else "Pegawai"
    st.markdown(f"""
    <div class="card">
      <h4>👤 Maklumat Pegawai</h4>
      <div class="chip-row">
        <span class="chip">👤 {nama_papar}</span>
        <span class="chip">🎓 {gred_pilih} — {gred['nama']}</span>
        <span class="chip">🗓️ Umur semasa: {umur_sekarang} tahun</span>
        <span class="chip">📅 Umur persaraan: {umur_persaraan} tahun</span>
        <span class="chip">⏳ Tahun mula: {tahun_mula}</span>
        <span class="chip-gold">📊 Tempoh perkhidmatan semasa: {tahun_khidmat_sekarang} tahun</span>
        <span class="chip-gold">📊 Tempoh persaraan: {tahun_khidmat_persaraan} tahun</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Main results ──────────────────────────────────────────────────────────
    st.markdown('<div class="sec-title">💰 Keputusan Ramalan Gaji</div>', unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        <div class="result-main">
          <div class="label">💵 Gaji Akhir Anggaran</div>
          <div class="amount">{format_rm(gaji_akhir)}</div>
          <div class="sub-amount">Semasa persaraan umur {umur_persaraan}</div>
        </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="result-main" style="background:linear-gradient(135deg,#065f46,#047857);border-color:#10b981;box-shadow:0 6px 30px rgba(16,185,129,0.4);">
          <div class="label" style="color:#6ee7b7;">🏦 Pencen Bulanan</div>
          <div class="amount">{format_rm(pencen_bulanan)}</div>
          <div class="sub-amount" style="color:#a7f3d0;">Kadar pencen: {kadar*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    with r3:
        st.markdown(f"""
        <div class="result-main" style="background:linear-gradient(135deg,#78350f,#b45309);border-color:#f59e0b;box-shadow:0 6px 30px rgba(245,158,11,0.4);">
          <div class="label" style="color:#fde68a;">🎁 GCR (Maks 90 hari)</div>
          <div class="amount">{format_rm(gcr)}</div>
          <div class="sub-amount" style="color:#fef3c7;">Gantian Cuti Rehat</div>
        </div>""", unsafe_allow_html=True)

    # ── Kenaikan gaji chart ───────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">📈 Unjuran Kenaikan Gaji Sehingga Persaraan</div>', unsafe_allow_html=True)

    umur_list = list(range(umur_sekarang + 1, umur_persaraan + 1))
    if umur_list and gaji_tahunan:
        fig, ax = plt.subplots(figsize=(10, 3.8))
        fig.patch.set_facecolor("#0a0e27")
        ax.set_facecolor("#0d1b4b")

        ax.plot(umur_list, gaji_tahunan, color="#3b82f6", linewidth=2.5, marker="o",
                markersize=5, markerfacecolor="#93c5fd")
        ax.fill_between(umur_list, gaji_tahunan, alpha=0.15, color="#3b82f6")

        ax.axhline(gred["gaji_max"], color="#f59e0b", linewidth=1.2,
                   linestyle="--", alpha=0.7, label=f"Gaji Max {gred_pilih}: {format_rm(gred['gaji_max'])}")
        ax.axhline(gaji_sekarang, color="#6ee7b7", linewidth=1, linestyle=":",
                   alpha=0.6, label=f"Gaji Semasa: {format_rm(gaji_sekarang)}")

        ax.set_xlabel("Umur (tahun)", color="#93c5fd", fontsize=10)
        ax.set_ylabel("Gaji (RM)", color="#93c5fd", fontsize=10)
        ax.tick_params(colors="#93c5fd", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1d4ed8")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"RM {x:,.0f}"))
        ax.legend(fontsize=8, facecolor="#0d1b4b", labelcolor="#bfdbfe",
                  edgecolor="#3b82f6", framealpha=0.8)

        # Annotate final salary
        ax.annotate(f"  {format_rm(gaji_akhir)}",
                    xy=(umur_list[-1], gaji_tahunan[-1]),
                    color="#fff", fontsize=9, fontweight="bold")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Increment detail table ────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">📊 Jadual Kenaikan Gaji Tahunan</div>', unsafe_allow_html=True)

    if umur_list and gaji_tahunan:
        rows = []
        prev = gaji_sekarang
        for i, (umur_t, gaji_t) in enumerate(zip(umur_list, gaji_tahunan)):
            delta = gaji_t - prev
            prev = gaji_t
            rows.append({
                "Tahun Ke": i + 1,
                "Umur": umur_t,
                "Tahun Calendar": tahun_sekarang + i + 1,
                "Gaji (RM)": gaji_t,
                "Kenaikan (RM)": delta,
                "Status": "🔒 Ceiling" if gaji_t >= gred["gaji_max"] else "📈 Naik"
            })

        df = pd.DataFrame(rows)
        df["Gaji (RM)"] = df["Gaji (RM)"].apply(lambda x: f"RM {x:,.2f}")
        df["Kenaikan (RM)"] = df["Kenaikan (RM)"].apply(lambda x: f"RM {x:,.2f}")
        df.set_index("Tahun Ke", inplace=True)
        st.dataframe(df, use_container_width=True)

    # ── Increment alert per grade ─────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">🔔 Makluman Kenaikan & Syarat Persaraan</div>', unsafe_allow_html=True)

    for g, d in GRED_DATA.items():
        is_current = g == gred_pilih
        border = d["color"] if is_current else "rgba(255,255,255,0.1)"
        bg = d["bg"] if is_current else "rgba(255,255,255,0.03)"
        highlight = "⭐ GRED SEMASA ANDA" if is_current else ""

        st.markdown(f"""
        <div class="card" style="border-color:{border};background:{bg};">
          <h4 style="color:{d['color']};">{g} — {d['nama']} &nbsp; <span style="font-size:0.75rem;color:#fde68a;">{highlight}</span></h4>
          <div style="color:#93c5fd;font-size:0.82rem;margin-bottom:8px;">👩‍🏫 {d['jawatan']}</div>
          <div class="chip-row">
            <span class="chip">💰 Gaji Min: {format_rm(d['gaji_min'])}</span>
            <span class="chip">💰 Gaji Max: {format_rm(d['gaji_max'])}</span>
            <span class="chip-green">📈 IKT: {format_rm(d['increment_base'])} / tahun</span>
            <span class="chip-gold">⬆️ Anjakan: {d['anjakan_count']}x × {format_rm(d['anjakan_amount'])}</span>
          </div>
          <div style="margin-top:8px;font-size:0.85rem;color:#bfdbfe;">
            🏛️ <b>Syarat persaraan:</b> {d['syarat_persaraan']}<br>
            📌 <b>Nota:</b> {d['nota']}<br>
            🚀 <b>Kenaikan pangkat:</b> {d['kenaikan_pangkat']}
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Summary box ───────────────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card" style="border-color:#3b82f6;">
      <h4>📋 Rumusan Lengkap — {nama_papar}</h4>
      <div class="chip-row">
        <span class="chip-gold">🎓 Gred: {gred_pilih}</span>
        <span class="chip">📅 Persaraan: Umur {umur_persaraan} ({tahun_persaraan})</span>
        <span class="chip">⏳ Tempoh: {tahun_khidmat_persaraan} tahun</span>
        <span class="chip-green">💵 Gaji Akhir: {format_rm(gaji_akhir)}</span>
        <span class="chip-green">🏦 Pencen/bulan: {format_rm(pencen_bulanan)}</span>
        <span class="chip-gold">🎁 GCR: {format_rm(gcr)}</span>
        <span class="chip">📊 Kadar Pencen: {kadar*100:.1f}%</span>
        <span class="chip">📈 Kenaikan/tahun: {format_rm(increment_tahunan)}</span>
      </div>
      <div style="margin-top:10px;color:#bfdbfe;font-size:0.88rem;">
        💡 Anggaran pendapatan pencen setahun: <b style="color:#6ee7b7;">{format_rm(pencen_bulanan * 12)}</b> &nbsp;|&nbsp;
        Jumlah GCR + Pencen tahun pertama: <b style="color:#fde68a;">{format_rm(gcr + pencen_bulanan * 12)}</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
      ⚠️ <b>Penafian:</b> Kiraan ini adalah anggaran berdasarkan Sistem Saraan Malaysia (SSM 2024) dan Akta Pencen 1980.
      Gaji sebenar bergantung kepada prestasi kerja, tanggungan, skim, dan pekeliling semasa.
      Sila rujuk <b>Jabatan Perkhidmatan Awam (JPA)</b>, <b>Pejabat Kewangan KPPK</b>, atau
      <b>Bahagian Gaji Jabatan Akauntan Negara Malaysia (JANM)</b> untuk pengesahan rasmi.
    </div>
    """, unsafe_allow_html=True)
