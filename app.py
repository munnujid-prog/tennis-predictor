"""
Landing Page — Multi-App Dashboard
Navigasi ke:
  - 🎾 ATP Match Win Probability Predictor
  - 🎓 Peramal Gaji Persaraan DH KPPK
Jalankan dengan:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Dashboard Utama",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
* { font-family: 'Inter', sans-serif; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0e27 0%, #0d1b4b 50%, #0f2027 100%);
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b4b 0%, #1a1a2e 100%);
    border-right: 2px solid #3b82f6;
}
[data-testid="stSidebar"] * { color: #bfdbfe !important; }
[data-testid="stSidebar"] hr { border-color: #3b82f6 !important; }

.landing-hero {
    background: linear-gradient(90deg, #1d4ed8 0%, #7c3aed 50%, #0f3460 100%);
    border-radius: 24px; padding: 48px 40px; margin-bottom: 36px;
    text-align: center; box-shadow: 0 10px 50px rgba(37,99,235,0.4);
    border: 1px solid rgba(147,197,253,0.2);
}
.landing-hero h1 {
    color: #fff !important; font-size: 2.8rem !important; font-weight: 900 !important;
    margin: 0 0 10px 0; letter-spacing: 0.5px;
}
.landing-hero p { color: rgba(255,255,255,0.85) !important; font-size: 1.1rem; margin: 0; }

.app-card {
    border-radius: 20px; padding: 32px 28px; text-align: center;
    box-shadow: 0 6px 30px rgba(0,0,0,0.4); transition: transform 0.2s;
    border: 2px solid rgba(255,255,255,0.1); margin-bottom: 16px;
}
.app-card-tennis {
    background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 60%, #16213e 100%);
    border-color: #e94560;
}
.app-card-salary {
    background: linear-gradient(135deg, #0d1b4b 0%, #1e3a8a 60%, #1d4ed8 100%);
    border-color: #3b82f6;
}
.app-card .icon { font-size: 3.5rem; margin-bottom: 12px; }
.app-card h2 { color: #fff !important; font-size: 1.5rem !important; font-weight: 800 !important; margin: 0 0 8px 0; }
.app-card p { color: rgba(255,255,255,0.75) !important; font-size: 0.92rem; margin: 0 0 20px 0; line-height: 1.6; }

.feature-list {
    text-align: left; margin: 0 0 20px 0; padding: 0; list-style: none;
}
.feature-list li { color: rgba(255,255,255,0.7) !important; font-size: 0.85rem;
                   padding: 4px 0; display: flex; align-items: center; gap: 8px; }

.badge-tennis { background: rgba(233,69,96,0.2); border: 1px solid #e94560;
                border-radius: 8px; padding: 4px 12px; font-size: 0.75rem; color: #ff8099; }
.badge-salary { background: rgba(59,130,246,0.2); border: 1px solid #3b82f6;
                border-radius: 8px; padding: 4px 12px; font-size: 0.75rem; color: #93c5fd; }

.divider-gradient {
    height: 2px;
    background: linear-gradient(90deg, #e94560, #7c3aed, #3b82f6);
    border-radius: 2px; margin: 32px 0;
}

.info-box {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px; padding: 20px 24px; margin-bottom: 16px;
}
.info-box h4 { color: #93c5fd !important; font-size: 0.9rem; letter-spacing: 1.5px;
               text-transform: uppercase; margin: 0 0 10px 0; }
.info-box p  { color: #bfdbfe !important; font-size: 0.88rem; margin: 0; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 Dashboard Utama")
    st.markdown("---")
    st.markdown("**Navigasi:**")
    st.markdown("- 🎾 Tennis Predictor")
    st.markdown("- 🎓 Salary Predictor")
    st.markdown("---")
    st.markdown("Gunakan menu di sebelah kiri untuk bertukar antara aplikasi.")
    st.markdown("---")
    st.caption("Versi 1.0 · 2024")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="landing-hero">
  <h1>🏠 Dashboard Utama</h1>
  <p>Pilih aplikasi di bawah atau gunakan menu navigasi di sebelah kiri</p>
</div>
""", unsafe_allow_html=True)

# ── App Cards ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="app-card app-card-tennis">
      <div class="icon">🎾</div>
      <h2>ATP Match Predictor</h2>
      <p>Ramalkan keputusan perlawanan tenis ATP menggunakan model Random Forest yang dilatih dengan 68,760 perlawanan sejarah.</p>
      <ul class="feature-list">
        <li>📊 Analisis perbezaan ranking ATP</li>
        <li>🌍 Sokongan 4 permukaan gelanggang</li>
        <li>🏆 Semua siri peringkat grand slam</li>
        <li>📈 Graf kebarangkalian menang visual</li>
      </ul>
      <span class="badge-tennis">Random Forest · 2000–2026</span>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/1_🎾_Tennis_Predictor.py", label="🎾 Buka Tennis Predictor", use_container_width=True)

with col2:
    st.markdown("""
    <div class="app-card app-card-salary">
      <div class="icon">🎓</div>
      <h2>Peramal Gaji Persaraan</h2>
      <p>Kira anggaran gaji persaraan untuk pensyarah KPPK Gred DH12–DH26 berdasarkan SSM 2024 dan Akta Pencen 1980.</p>
      <ul class="feature-list">
        <li>💰 Unjuran gaji sehingga persaraan</li>
        <li>🏦 Kiraan pencen bulanan & GCR</li>
        <li>📋 Jadual kenaikan gaji tahunan</li>
        <li>🔔 Syarat & makluman kenaikan pangkat</li>
      </ul>
      <span class="badge-salary">SSM 2024 · Akta Pencen 1980</span>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/2_🎓_Salary_Predictor.py", label="🎓 Buka Salary Predictor", use_container_width=True)

# ── Info Section ──────────────────────────────────────────────────────────────
st.markdown('<div class="divider-gradient"></div>', unsafe_allow_html=True)

info1, info2, info3 = st.columns(3)
with info1:
    st.markdown("""
    <div class="info-box">
      <h4>🎾 Tentang Tennis Predictor</h4>
      <p>Model Random Forest dilatih dengan data ATP 2000–2018. 
      Menggunakan ciri-ciri ranking, permukaan, siri, dan format perlawanan 
      untuk meramalkan kebarangkalian menang.</p>
    </div>
    """, unsafe_allow_html=True)

with info2:
    st.markdown("""
    <div class="info-box">
      <h4>🎓 Tentang Salary Predictor</h4>
      <p>Berasaskan Pekeliling Perkhidmatan Bil. 1/2024 (SSM 2024) 
      dan Akta Pencen 1980. Meliputi gred DH12, DH14, DH22, dan DH26 
      bagi pensyarah Politeknik & Kolej Komuniti.</p>
    </div>
    """, unsafe_allow_html=True)

with info3:
    st.markdown("""
    <div class="info-box">
      <h4>ℹ️ Cara Penggunaan</h4>
      <p>Klik butang di atas atau gunakan menu navigasi di sebelah kiri 
      skrin untuk bertukar antara aplikasi. Setiap aplikasi berdiri sendiri 
      dengan tetapan dan data tersendiri.</p>
    </div>
    """, unsafe_allow_html=True)
