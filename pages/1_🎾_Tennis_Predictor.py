"""
Page 1 — ATP Match Win Probability Predictor
Multi-page Streamlit app
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import OneHotEncoder

# -- Paths ------------------------------------------------------------------
MDIR = "models"
PROC = "processed"

ROUND_MAP = {
    "1st Round":    1,
    "2nd Round":    2,
    "3rd Round":    3,
    "Round of 16":  4,
    "Quarterfinals":5,
    "Semifinals":   6,
    "The Final":    7,
}
CAT_COLS = ["Surface", "Court", "Series"]

SURFACE_EMOJI = {"Hard": "💙", "Clay": "🧡", "Grass": "💚", "Carpet": "💜"}
SURFACE_COLOR = {"Hard": "#1E90FF", "Clay": "#E07040", "Grass": "#2ECC71", "Carpet": "#9B59B6"}
ROUND_EMOJI   = {"1st Round":"1️⃣","2nd Round":"2️⃣","3rd Round":"3️⃣",
                 "Round of 16":"🔢","Quarterfinals":"🎯","Semifinals":"⚡","The Final":"🏆"}

# -- Page config ------------------------------------------------------------
st.set_page_config(
    page_title="ATP Match Predictor",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Custom CSS -------------------------------------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
    border-right: 2px solid #e94560;
}
[data-testid="stSidebar"] * { color: #eaeaea !important; }
[data-testid="stSidebar"] h1 { color: #FFD700 !important; font-size: 1.4rem !important; }
[data-testid="stSidebar"] hr { border-color: #e94560 !important; }

.hero-banner {
    background: linear-gradient(90deg, #e94560 0%, #f5a623 50%, #e94560 100%);
    border-radius: 16px; padding: 28px 36px; margin-bottom: 24px;
    text-align: center; box-shadow: 0 8px 32px rgba(233,69,96,0.4);
}
.hero-banner h1 {
    color: #fff !important; font-size: 2.6rem !important; font-weight: 900 !important;
    letter-spacing: 1px; margin: 0; text-shadow: 2px 2px 8px rgba(0,0,0,0.4);
}
.hero-banner p { color: rgba(255,255,255,0.9) !important; font-size: 1.05rem; margin: 8px 0 0 0; }

.player-card { border-radius: 14px; padding: 18px 20px 14px 20px; margin-bottom: 8px;
               border: 2px solid rgba(255,255,255,0.12); box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
.card-p1 { background: linear-gradient(135deg, #1a1a4e 0%, #2d2d8f 100%); }
.card-p2 { background: linear-gradient(135deg, #4e1a1a 0%, #8f2d2d 100%); }
.player-label { font-size: 0.78rem; font-weight: 700; letter-spacing: 2px;
                text-transform: uppercase; margin-bottom: 4px; }
.label-p1 { color: #7EB8FF; }
.label-p2 { color: #FF8C8C; }
.rank-badge { display: inline-block; background: rgba(255,255,255,0.15); border-radius: 20px;
              padding: 2px 12px; font-size: 0.82rem; color: #FFD700; font-weight: 700; margin-top: 4px; }

.context-card { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);
                border-radius: 14px; padding: 18px 20px; margin-bottom: 16px; }
.context-card h4 { color: #FFD700 !important; font-size: 0.85rem; letter-spacing: 2px;
                   text-transform: uppercase; margin: 0 0 12px 0; }

.section-divider { height: 3px;
    background: linear-gradient(90deg, #e94560, #f5a623, #2ECC71, #1E90FF, #e94560);
    border-radius: 2px; margin: 20px 0; }

.result-winner { background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 100%);
    border: 2px solid #52b788; border-radius: 16px; padding: 20px 24px;
    text-align: center; margin-bottom: 16px; box-shadow: 0 4px 24px rgba(82,183,136,0.35); }
.result-winner h2 { color: #b7e4c7 !important; font-size: 0.85rem; letter-spacing: 2px;
    text-transform: uppercase; margin: 0 0 6px 0; }
.result-winner h1 { color: #fff !important; font-size: 2rem !important; font-weight: 900 !important; margin: 0; }
.result-winner .trophy { font-size: 2.4rem; }

.prob-card { border-radius: 14px; padding: 16px 20px; text-align: center; border: 2px solid rgba(255,255,255,0.12); }
.prob-card-win  { background: linear-gradient(135deg,#1a472a,#2d6a4f); border-color:#52b788; }
.prob-card-lose { background: linear-gradient(135deg,#1a1a2e,#2c2c4e); border-color:#4a4a7a; }
.prob-pct { font-size: 2.4rem; font-weight: 900; line-height: 1; }
.prob-pct-win  { color: #b7e4c7; }
.prob-pct-lose { color: #a8a8c0; }
.prob-name { font-size: 0.95rem; color: #ccc; margin-top: 4px; }
.prob-rank { font-size: 0.78rem; color: #FFD700; margin-top: 2px; }

.stat-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 12px; }
.stat-chip { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.18);
             border-radius: 20px; padding: 4px 14px; font-size: 0.82rem; color: #eee; }

[data-testid="stSelectbox"] label,
[data-testid="stRadio"] label,
[data-testid="stNumberInput"] label { color: #ccc !important; font-size: 0.85rem !important; }

div[data-baseweb="select"] > div { background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.2) !important; color: #fff !important; border-radius: 8px !important; }
div[data-baseweb="select"] span { color: #fff !important; }

[data-baseweb="popover"], [data-baseweb="popover"] * { background: #ffffff !important; color: #111111 !important; }
[data-baseweb="menu"],    [data-baseweb="menu"] *    { background: #ffffff !important; color: #111111 !important; }
[role="option"], [role="option"] * { background: #ffffff !important; color: #111111 !important; }
[role="option"]:hover, [role="option"]:hover * { background: #e8f0fe !important; color: #000000 !important; }

[data-testid="stButton"] > button {
    background: linear-gradient(90deg, #e94560 0%, #f5a623 100%) !important;
    color: #fff !important; font-size: 1.1rem !important; font-weight: 700 !important;
    border: none !important; border-radius: 10px !important; padding: 12px 48px !important;
    width: 100% !important; box-shadow: 0 4px 16px rgba(233,69,96,0.45) !important;
}
[data-testid="stButton"] > button:hover { box-shadow: 0 6px 24px rgba(233,69,96,0.6) !important; }

[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] p { color: #ddd; }
h2, h3 { color: #FFD700 !important; }
</style>
""", unsafe_allow_html=True)

# -- Cached resource loading ------------------------------------------------
@st.cache_resource
def load_artefacts():
    model        = joblib.load(os.path.join(MDIR, "rf_model.pkl"))
    encoder      = joblib.load(os.path.join(MDIR, "encoder.pkl"))
    feature_cols = joblib.load(os.path.join(MDIR, "feature_columns.pkl"))
    return model, encoder, feature_cols

@st.cache_data
def load_player_lookup():
    train = pd.read_csv(os.path.join(PROC, "train.csv"), parse_dates=["Date"])
    p1 = train[["Date","Player_1","Rank_1"]].rename(columns={"Player_1":"player","Rank_1":"rank"})
    p2 = train[["Date","Player_2","Rank_2"]].rename(columns={"Player_2":"player","Rank_2":"rank"})
    all_players = pd.concat([p1, p2], ignore_index=True).dropna(subset=["rank"])
    return (all_players.sort_values("Date")
            .groupby("player")["rank"].last().astype(int).to_dict())

@st.cache_data
def get_series_options():
    train = pd.read_csv(os.path.join(PROC, "train.csv"))
    return sorted(train["Series"].dropna().unique().tolist())

# -- Load artefacts ---------------------------------------------------------
try:
    model, encoder, feature_cols = load_artefacts()
    player_lookup  = load_player_lookup()
    series_options = get_series_options()
except FileNotFoundError:
    st.error("Model artefacts not found. Run the pipeline first:\n\n"
             "`py src/01_prepare.py` → `py src/02_features.py` → `py src/03_train.py`")
    st.stop()

player_names = sorted(player_lookup.keys())

# -- Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎾 ATP Predictor")
    st.markdown("---")
    st.markdown("**Model:** Random Forest")
    st.markdown("**Training data:** 2000 – 2018")
    st.markdown("**Players available:** {:,}".format(len(player_names)))
    st.markdown("---")
    st.markdown("**Features used:**")
    st.markdown("- 📊 ATP ranking difference & ratio\n- 🌍 Surface & Court type\n- 🏆 Tournament series\n- 🔢 Round & Match format")
    st.markdown("---")
    st.caption("⚠️ Predictions use ranking and match-context features only. "
               "Player names and betting odds are NOT used by the model.")

# -- Hero banner ------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
  <h1>🎾 ATP Match Win Probability Predictor</h1>
  <p>Powered by Random Forest · Historical ATP data 2000–2026 · 68,760 matches</p>
</div>
""", unsafe_allow_html=True)

# -- Player selection -------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="player-card card-p1"><div class="player-label label-p1">🔵 Player 1</div>', unsafe_allow_html=True)
    player1 = st.selectbox("Select Player 1", player_names, key="p1", label_visibility="collapsed")
    r1_default = player_lookup.get(player1)
    st.markdown(f'<div class="rank-badge">ATP Rank #{r1_default if r1_default else "?"}</div></div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="player-card card-p2"><div class="player-label label-p2">🔴 Player 2</div>', unsafe_allow_html=True)
    player2 = st.selectbox("Select Player 2", player_names, index=min(1, len(player_names)-1), key="p2", label_visibility="collapsed")
    r2_default = player_lookup.get(player2)
    st.markdown(f'<div class="rank-badge">ATP Rank #{r2_default if r2_default else "?"}</div></div>', unsafe_allow_html=True)

# -- Manual rank override ---------------------------------------------------
if r1_default is None:
    st.warning(f"**{player1}** not in training data.")
    r1_default = st.number_input(f"Enter rank for {player1}", 1, 2000, 100)
if r2_default is None:
    st.warning(f"**{player2}** not in training data.")
    r2_default = st.number_input(f"Enter rank for {player2}", 1, 2000, 100)

# -- Match context ----------------------------------------------------------
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="context-card"><h4>⚙️ Match Context</h4>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    surface = st.selectbox("🌍 Surface", ["Hard","Clay","Grass","Carpet"])
with c2:
    round_  = st.selectbox("🏟️ Round", list(ROUND_MAP.keys()))
with c3:
    series  = st.selectbox("🏆 Series", series_options)
with c4:
    court   = st.selectbox("🔆 Court", ["Outdoor","Indoor"])

best_of = st.radio("🎯 Best of", [3, 5], horizontal=True)
st.markdown('</div>', unsafe_allow_html=True)

# -- Predict button ---------------------------------------------------------
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
predict_clicked = st.button("🎾 PREDICT WINNER")

# -- Prediction logic -------------------------------------------------------
if predict_clicked:
    if player1 == player2:
        st.error("⚠️ Please select two different players.")
    else:
        rank1 = int(r1_default)
        rank2 = int(r2_default)

        rank_diff  = rank1 - rank2
        rank_ratio = rank1 / (rank2 + 1e-9)
        round_num  = ROUND_MAP[round_]

        cat_input = pd.DataFrame([[surface, court, series]], columns=CAT_COLS)
        cat_enc   = encoder.transform(cat_input)
        cat_names = encoder.get_feature_names_out(CAT_COLS)

        num_part = np.array([[rank_diff, rank_ratio, round_num, best_of]])
        full_vec = np.concatenate([num_part, cat_enc], axis=1)

        full_df = pd.DataFrame(full_vec,
            columns=["rank_diff","rank_ratio","round_num","best_of"] + list(cat_names))
        for col in feature_cols:
            if col not in full_df.columns:
                full_df[col] = 0.0
        X_input = full_df[feature_cols].values

        probas = model.predict_proba(X_input)[0]
        p1_win = probas[1]
        p2_win = probas[0]

        winner      = player1 if p1_win >= 0.5 else player2
        winner_rank = rank1   if p1_win >= 0.5 else rank2
        winner_prob = max(p1_win, p2_win)

        surf_color = SURFACE_COLOR.get(surface, "#888")
        surf_emoji = SURFACE_EMOJI.get(surface, "🎾")
        rnd_emoji  = ROUND_EMOJI.get(round_, "🎾")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="result-winner">
          <div class="trophy">🏆</div>
          <h2>Predicted Winner</h2>
          <h1>{winner}</h1>
          <div style="margin-top:8px;font-size:1.2rem;color:#FFD700;font-weight:700;">
            {winner_prob*100:.1f}% win probability · ATP Rank #{winner_rank}
          </div>
          <div style="margin-top:10px;" class="stat-row" style="justify-content:center;">
            <span class="stat-chip">{surf_emoji} {surface}</span>
            <span class="stat-chip">{rnd_emoji} {round_}</span>
            <span class="stat-chip">🏆 {series}</span>
            <span class="stat-chip">🎯 Best of {best_of}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        res_col1, res_col2 = st.columns(2)
        p1_class = "prob-card-win"  if p1_win >= 0.5 else "prob-card-lose"
        p2_class = "prob-card-win"  if p2_win >  p1_win else "prob-card-lose"
        p1_pct_class = "prob-pct-win"  if p1_win >= 0.5 else "prob-pct-lose"
        p2_pct_class = "prob-pct-win"  if p2_win >  p1_win else "prob-pct-lose"

        with res_col1:
            st.markdown(f"""
            <div class="prob-card {p1_class}">
              <div class="prob-pct {p1_pct_class}">{p1_win*100:.1f}%</div>
              <div class="prob-name">{"🏆 " if p1_win >= 0.5 else ""}{player1}</div>
              <div class="prob-rank">ATP Rank #{rank1}</div>
            </div>""", unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
            <div class="prob-card {p2_class}">
              <div class="prob-pct {p2_pct_class}">{p2_win*100:.1f}%</div>
              <div class="prob-name">{"🏆 " if p2_win > p1_win else ""}{player2}</div>
              <div class="prob-rank">ATP Rank #{rank2}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(9, 2.8))
        fig.patch.set_facecolor("#0f2027")
        ax.set_facecolor("#0f2027")

        colors  = ["#52b788" if p1_win >= 0.5 else "#4a4a7a",
                   "#52b788" if p2_win > p1_win else "#4a4a7a"]
        players = [f"{player1}\n(#{rank1})", f"{player2}\n(#{rank2})"]
        probs   = [p1_win * 100, p2_win * 100]

        bars = ax.barh(players, probs, color=colors, height=0.45, edgecolor=(1, 1, 1, 0.1))
        ax.set_xlim(0, 100)
        ax.set_xlabel("Win Probability (%)", color="#aaa", fontsize=10)
        ax.tick_params(colors="#ccc", labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.xaxis.label.set_color("#aaa")

        for bar, prob in zip(bars, probs):
            ax.text(min(prob + 1.5, 93), bar.get_y() + bar.get_height() / 2,
                    f"{prob:.1f}%", va="center", fontsize=13, fontweight="bold", color="#fff")

        ax.axvline(50, color=surf_color, linewidth=1.5, linestyle="--", alpha=0.6)
        ax.text(50.8, 1.08, "50%", color=surf_color, fontsize=8, alpha=0.8,
                transform=ax.get_xaxis_transform())

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown(f"""
        <div style="margin-top:8px;display:flex;gap:10px;flex-wrap:wrap;">
          <span class="stat-chip" style="border-color:{surf_color};color:{surf_color};">{surf_emoji} {surface} Court</span>
          <span class="stat-chip">{rnd_emoji} {round_}</span>
          <span class="stat-chip">🏟️ {court}</span>
          <span class="stat-chip">🏆 {series}</span>
          <span class="stat-chip">🎯 Best of {best_of}</span>
          <span class="stat-chip">📊 Rank gap: {abs(rank1-rank2)}</span>
        </div>
        """, unsafe_allow_html=True)
