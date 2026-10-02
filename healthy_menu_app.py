"""
Healthy Food Menu Predictor
For mothers managing family nutrition and patients with critical illnesses.
Run with:  streamlit run healthy_menu_app.py
"""

import streamlit as st
import pandas as pd

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthy Menu Predictor",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 50%, #f0f9ff 100%);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #064e3b 0%, #065f46 60%, #047857 100%);
    border-right: 3px solid #10b981;
}
[data-testid="stSidebar"] * { color: #d1fae5 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2 { color: #6ee7b7 !important; }
[data-testid="stSidebar"] hr { border-color: #10b981 !important; }

.hero {
    background: linear-gradient(90deg, #059669 0%, #10b981 40%, #34d399 100%);
    border-radius: 18px;
    padding: 28px 36px;
    margin-bottom: 24px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(16,185,129,0.3);
}
.hero h1 { color: #fff !important; font-size: 2.4rem !important; font-weight: 900 !important; margin: 0; }
.hero p  { color: rgba(255,255,255,0.92) !important; font-size: 1.05rem; margin: 8px 0 0 0; }

.section-title {
    font-size: 1.1rem; font-weight: 700; color: #065f46;
    border-left: 4px solid #10b981; padding-left: 10px;
    margin: 20px 0 12px 0;
}
.condition-badge {
    display: inline-block; border-radius: 20px; padding: 4px 14px;
    font-size: 0.82rem; font-weight: 700; margin: 3px;
}
.meal-card {
    background: #fff; border-radius: 14px; padding: 16px 18px;
    border: 2px solid #d1fae5; margin-bottom: 12px;
    box-shadow: 0 2px 12px rgba(16,185,129,0.1);
}
.meal-card h4 { color: #065f46 !important; margin: 0 0 8px 0; font-size: 1rem; }
.meal-item { color: #1f2937; font-size: 0.93rem; margin: 3px 0; }
.nutrient-row {
    display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px;
}
.nutrient-chip {
    background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 12px;
    padding: 3px 10px; font-size: 0.78rem; color: #065f46; font-weight: 600;
}
.safe-food { color: #065f46; font-weight: 600; }
.avoid-food { color: #dc2626; font-weight: 600; }
.warning-box {
    background: #fef3c7; border: 2px solid #f59e0b; border-radius: 12px;
    padding: 14px 18px; margin: 16px 0; color: #92400e;
}
.info-box {
    background: #eff6ff; border: 2px solid #93c5fd; border-radius: 12px;
    padding: 14px 18px; margin: 16px 0; color: #1e40af;
}
.divider {
    height: 3px;
    background: linear-gradient(90deg, #10b981, #34d399, #6ee7b7, #10b981);
    border-radius: 2px; margin: 20px 0;
}
/* Dropdown popup fix */
[data-baseweb="popover"], [data-baseweb="popover"] * { background: #fff !important; color: #111 !important; }
[data-baseweb="menu"],    [data-baseweb="menu"] *    { background: #fff !important; color: #111 !important; }
[role="option"], [role="option"] * { background: #fff !important; color: #111 !important; }
[role="option"]:hover, [role="option"]:hover * { background: #ecfdf5 !important; color: #000 !important; }
div[data-baseweb="select"] > div { background: #fff !important; border: 1px solid #a7f3d0 !important; color: #111 !important; border-radius: 8px !important; }
div[data-baseweb="select"] span { color: #111 !important; }
[data-testid="stButton"] > button {
    background: linear-gradient(90deg, #059669, #10b981) !important;
    color: #fff !important; font-size: 1.05rem !important; font-weight: 700 !important;
    border: none !important; border-radius: 10px !important; padding: 10px 36px !important;
    width: 100% !important; box-shadow: 0 4px 14px rgba(16,185,129,0.4) !important;
}
[data-testid="stMultiSelect"] label { color: #065f46 !important; font-weight: 600 !important; }
[data-testid="stCheckbox"] label { color: #1f2937 !important; }
</style>
""", unsafe_allow_html=True)

# ── Knowledge Base ───────────────────────────────────────────────────────────
CONDITIONS = {
    "Diabetes": {
        "color": "#7c3aed", "bg": "#ede9fe", "emoji": "🩸",
        "description": "Focus on low-GI foods, controlled carbohydrates, high fibre.",
        "safe": ["Leafy greens (spinach, kale)", "Broccoli, cauliflower", "Brown rice (small portion)", "Oats", "Lentils & chickpeas", "Fish (salmon, sardines)", "Chicken breast (no skin)", "Eggs", "Tofu & tempeh", "Avocado", "Berries (strawberry, blueberry)", "Apple (small)", "Nuts (almonds, walnuts)", "Olive oil", "Plain low-fat yogurt"],
        "avoid": ["White rice (large portion)", "White bread & white flour", "Sugary drinks & juices", "Candy, cake, cookies", "Fried foods", "Full-fat dairy", "Processed meats (hot dog, sausage)", "Alcohol", "Honey & syrup in excess", "Potato chips"],
        "calories": (1400, 1800), "carbs_pct": 45, "protein_pct": 25, "fat_pct": 30,
    },
    "Hypertension": {
        "color": "#dc2626", "bg": "#fee2e2", "emoji": "❤️",
        "description": "DASH diet: low sodium, high potassium, low saturated fat.",
        "safe": ["Banana", "Watermelon", "Berries", "Leafy greens", "Sweet potato", "Oats", "Brown rice", "Fish (salmon, mackerel)", "Chicken breast", "Low-fat dairy (yogurt, milk)", "Nuts & seeds", "Beans & lentils", "Garlic & herbs (as seasoning)", "Olive oil", "Unsalted nuts"],
        "avoid": ["Table salt & salty condiments", "Processed & canned food", "Pickles & preserved foods", "Fast food & instant noodles", "Fried chicken & fried fish", "Full-fat cheese", "Alcohol", "Caffeine in excess", "Salted snacks", "Red meat (large portions)"],
        "calories": (1600, 2000), "carbs_pct": 50, "protein_pct": 20, "fat_pct": 30,
    },
    "Heart Disease": {
        "color": "#e11d48", "bg": "#ffe4e6", "emoji": "💗",
        "description": "Heart-healthy diet: low saturated fat, no trans fat, omega-3 rich.",
        "safe": ["Fatty fish (salmon, tuna, sardines)", "Oats & whole grains", "Flaxseeds & chia seeds", "Walnuts & almonds", "Olive oil", "Avocado", "Dark leafy greens", "Beans & legumes", "Berries & fruits", "Low-fat dairy", "Garlic & turmeric", "Dark chocolate (>70%)", "Green tea", "Tomatoes", "Broccoli"],
        "avoid": ["Trans fats (margarine, shortening)", "Saturated fat (butter, lard)", "Red meat (fatty cuts)", "Full-fat dairy", "Fried foods", "Sugary desserts", "Processed meats", "Alcohol", "Salt in excess", "Sugary beverages"],
        "calories": (1500, 1900), "carbs_pct": 50, "protein_pct": 20, "fat_pct": 30,
    },
    "Kidney Disease": {
        "color": "#d97706", "bg": "#fef3c7", "emoji": "🫘",
        "description": "Low potassium, low phosphorus, controlled protein, low sodium.",
        "safe": ["White rice", "White bread (low phosphorus)", "Cauliflower", "Cabbage", "Green beans", "Apple", "Grapes", "Strawberries", "Egg whites", "Chicken breast (small portion)", "White fish", "Unsalted butter (small amount)", "Olive oil", "Rice milk", "Blueberries"],
        "avoid": ["Bananas & oranges (high potassium)", "Tomatoes & potatoes (high potassium)", "Nuts & seeds (high phosphorus)", "Dairy products (high phosphorus)", "Whole grains (high phosphorus)", "Dark colas", "Processed foods (high sodium/phosphorus)", "Beans & lentils in large amounts", "Leafy greens (large amounts)", "Salt substitutes (contain potassium)"],
        "calories": (1800, 2200), "carbs_pct": 60, "protein_pct": 15, "fat_pct": 25,
    },
    "Cancer": {
        "color": "#7e22ce", "bg": "#f3e8ff", "emoji": "🎗️",
        "description": "Anti-inflammatory, antioxidant-rich, immune-boosting diet.",
        "safe": ["Berries (blueberry, raspberry)", "Cruciferous vegetables (broccoli, cabbage)", "Leafy greens (spinach, kale)", "Garlic & onion", "Turmeric & ginger", "Green tea", "Tomatoes (lycopene)", "Legumes & beans", "Whole grains", "Fatty fish (omega-3)", "Nuts & seeds", "Olive oil", "Carrots & sweet potato", "Citrus fruits (vitamin C)", "Yogurt (probiotics)"],
        "avoid": ["Processed & red meats", "Alcohol", "Sugary drinks & foods", "Fried & charred foods", "Trans fats", "Refined carbohydrates", "Excess salt", "Canned foods (BPA)", "Artificial additives & preservatives", "Mouldy food (aflatoxin)"],
        "calories": (2000, 2500), "carbs_pct": 50, "protein_pct": 25, "fat_pct": 25,
    },
}

MEAL_PLANS = {
    "general": {
        "breakfast": {"name": "Oat Porridge with Fruits", "items": ["1 cup rolled oats (cooked)", "1 small banana or ½ cup berries", "1 tsp honey (optional)", "1 glass low-fat milk or soy milk"], "cal": 350, "carb": 58, "protein": 12, "fat": 6},
        "snack_am": {"name": "Nuts & Fruit", "items": ["12 almonds or walnuts", "1 small apple or pear"], "cal": 180, "carb": 22, "protein": 4, "fat": 9},
        "lunch": {"name": "Brown Rice with Steamed Fish & Vegetables", "items": ["¾ cup brown rice", "1 palm-size steamed salmon or white fish", "1 cup mixed vegetables (broccoli, carrot, cabbage)", "1 tsp olive oil & garlic"], "cal": 480, "carb": 55, "protein": 32, "fat": 12},
        "snack_pm": {"name": "Yogurt with Seeds", "items": ["1 cup plain low-fat yogurt", "1 tbsp chia seeds or flaxseeds", "A few strawberries"], "cal": 170, "carb": 18, "protein": 10, "fat": 5},
        "dinner": {"name": "Chicken Soup with Vegetables", "items": ["1 palm-size chicken breast (no skin)", "1 cup mixed vegetables (spinach, carrot, potato)", "½ cup white rice or 1 slice wholemeal bread", "Low-sodium broth"], "cal": 380, "carb": 38, "protein": 30, "fat": 8},
    },
    "Diabetes": {
        "breakfast": {"name": "Egg & Vegetable Omelette", "items": ["2 eggs (scrambled/omelette)", "½ cup spinach, mushroom, tomato", "1 slice wholemeal toast (no butter)", "Unsweetened green tea"], "cal": 290, "carb": 22, "protein": 18, "fat": 12},
        "snack_am": {"name": "Almonds & Cucumber", "items": ["10 almonds", "½ cucumber (sliced)", "Pinch of black pepper"], "cal": 130, "carb": 8, "protein": 5, "fat": 9},
        "lunch": {"name": "Grilled Chicken with Quinoa & Salad", "items": ["1 palm-size grilled chicken breast", "½ cup cooked quinoa", "Large salad (lettuce, tomato, cucumber)", "1 tbsp olive oil + lemon dressing"], "cal": 420, "carb": 35, "protein": 38, "fat": 11},
        "snack_pm": {"name": "Plain Yogurt with Berries", "items": ["¾ cup plain Greek yogurt (unsweetened)", "½ cup blueberries or strawberries"], "cal": 140, "carb": 16, "protein": 12, "fat": 2},
        "dinner": {"name": "Steamed Fish with Vegetables & Small Rice", "items": ["1 palm-size steamed white fish or tofu", "1 cup stir-fried vegetables (no sauce)", "⅓ cup brown rice", "Garlic & ginger seasoning"], "cal": 350, "carb": 30, "protein": 28, "fat": 8},
    },
    "Hypertension": {
        "breakfast": {"name": "Oats with Banana & Low-fat Milk", "items": ["1 cup oats (cooked, no salt)", "1 small banana", "1 glass low-fat milk", "Cinnamon sprinkle"], "cal": 370, "carb": 62, "protein": 13, "fat": 6},
        "snack_am": {"name": "Watermelon or Fresh Fruit", "items": ["2 cups watermelon cubes", "or 1 orange / pear"], "cal": 90, "carb": 22, "protein": 2, "fat": 0},
        "lunch": {"name": "DASH Bowl — Brown Rice, Beans & Greens", "items": ["¾ cup brown rice", "½ cup cooked beans (no salt)", "1 cup steamed leafy greens", "1 tbsp olive oil", "Garlic & herbs seasoning (no salt)"], "cal": 460, "carb": 68, "protein": 16, "fat": 10},
        "snack_pm": {"name": "Unsalted Nuts", "items": ["Handful (15g) unsalted almonds or walnuts", "1 glass plain water or herbal tea"], "cal": 100, "carb": 3, "protein": 4, "fat": 9},
        "dinner": {"name": "Grilled Fish with Sweet Potato & Salad", "items": ["1 palm-size grilled mackerel or salmon", "½ medium sweet potato (baked)", "Mixed salad with olive oil & lemon", "No added salt — use herbs"], "cal": 420, "carb": 38, "protein": 30, "fat": 13},
    },
    "Heart Disease": {
        "breakfast": {"name": "Oats with Flaxseeds & Berries", "items": ["1 cup rolled oats", "1 tbsp ground flaxseeds", "½ cup mixed berries", "1 glass soy milk or low-fat milk"], "cal": 380, "carb": 58, "protein": 14, "fat": 9},
        "snack_am": {"name": "Walnuts & Green Tea", "items": ["12 walnut halves", "1 cup unsweetened green tea"], "cal": 130, "carb": 4, "protein": 4, "fat": 12},
        "lunch": {"name": "Tuna Salad with Wholemeal Bread", "items": ["½ can tuna in water (drained)", "Large salad (lettuce, tomato, avocado)", "2 slices wholemeal bread", "1 tsp olive oil & vinegar dressing"], "cal": 450, "carb": 42, "protein": 32, "fat": 14},
        "snack_pm": {"name": "Dark Chocolate & Almonds", "items": ["2 squares dark chocolate (>70%)", "8 almonds"], "cal": 150, "carb": 10, "protein": 4, "fat": 11},
        "dinner": {"name": "Baked Salmon with Broccoli & Quinoa", "items": ["1 palm-size baked salmon", "1 cup steamed broccoli with garlic", "½ cup cooked quinoa", "Lemon & herbs"], "cal": 430, "carb": 30, "protein": 38, "fat": 15},
    },
    "Kidney Disease": {
        "breakfast": {"name": "White Rice Congee with Egg White", "items": ["¾ cup plain white rice congee", "2 egg whites (boiled or poached)", "Small amount of unsalted butter", "½ cup green beans"], "cal": 280, "carb": 48, "protein": 14, "fat": 4},
        "snack_am": {"name": "Apple & Grapes", "items": ["1 small apple", "½ cup grapes", "Plain water"], "cal": 100, "carb": 26, "protein": 0, "fat": 0},
        "lunch": {"name": "White Rice with Chicken & Cabbage", "items": ["¾ cup white rice", "1 small chicken breast (boiled, no skin, no salt)", "1 cup boiled cabbage (low potassium)", "1 tsp olive oil"], "cal": 420, "carb": 58, "protein": 28, "fat": 8},
        "snack_pm": {"name": "Blueberries", "items": ["½ cup blueberries", "Plain rice milk (½ cup)"], "cal": 90, "carb": 18, "protein": 1, "fat": 1},
        "dinner": {"name": "White Fish with Cauliflower & Rice", "items": ["1 palm-size white fish (boiled/steamed, no salt)", "1 cup cauliflower (boiled)", "½ cup white rice", "Olive oil & herbs"], "cal": 370, "carb": 42, "protein": 26, "fat": 9},
    },
    "Cancer": {
        "breakfast": {"name": "Berry Smoothie Bowl", "items": ["½ cup blueberries + ½ cup raspberries", "1 cup Greek yogurt (probiotic)", "1 tbsp chia seeds", "1 tsp turmeric powder", "1 tsp ground flaxseed"], "cal": 320, "carb": 40, "protein": 18, "fat": 8},
        "snack_am": {"name": "Green Tea & Walnuts", "items": ["1 cup green tea (antioxidant)", "10 walnut halves"], "cal": 130, "carb": 4, "protein": 4, "fat": 12},
        "lunch": {"name": "Anti-Cancer Bowl", "items": ["½ cup cooked lentils or chickpeas", "1 cup steamed broccoli + cabbage", "½ cup brown rice", "1 tbsp olive oil", "Garlic, turmeric, ginger seasoning"], "cal": 450, "carb": 58, "protein": 20, "fat": 10},
        "snack_pm": {"name": "Carrot & Hummus", "items": ["1 medium carrot (sliced)", "3 tbsp hummus", "A few cherry tomatoes"], "cal": 160, "carb": 18, "protein": 6, "fat": 7},
        "dinner": {"name": "Baked Salmon with Spinach & Garlic", "items": ["1 palm-size baked salmon (omega-3)", "1 cup sautéed spinach with garlic & olive oil", "½ cup quinoa or brown rice", "Turmeric & lemon seasoning"], "cal": 460, "carb": 32, "protein": 36, "fat": 16},
    },
}

MEAL_LABELS = {
    "breakfast": ("🌅", "Breakfast"),
    "snack_am":  ("🍎", "Morning Snack"),
    "lunch":     ("🍽️", "Lunch"),
    "snack_pm":  ("🥜", "Afternoon Snack"),
    "dinner":    ("🌙", "Dinner"),
}

CONDITION_COLORS = {
    "Diabetes":      ("#7c3aed", "#ede9fe"),
    "Hypertension":  ("#dc2626", "#fee2e2"),
    "Heart Disease": ("#e11d48", "#ffe4e6"),
    "Kidney Disease":("#d97706", "#fef3c7"),
    "Cancer":        ("#7e22ce", "#f3e8ff"),
}

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🥗 Healthy Menu")
    st.markdown("---")
    st.markdown("**Designed for:**")
    st.markdown("- 👩 Mothers & families\n- 🏥 Patients with critical illness")
    st.markdown("---")
    st.markdown("**Conditions covered:**")
    for name, info in CONDITIONS.items():
        st.markdown(f"{info['emoji']} **{name}**")
    st.markdown("---")
    st.caption("⚠️ This app provides general dietary guidance only. "
               "Always consult your doctor or dietitian for personalised medical nutrition therapy.")

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🥗 Healthy Food Menu Predictor</h1>
  <p>Personalised daily meal plans · Safe & avoid food guide · Full nutrition info · Critical illness support</p>
</div>
""", unsafe_allow_html=True)

# ── User Profile ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">👤 Who is this meal plan for?</div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)
with col_a:
    user_type = st.selectbox("User type", ["Myself (healthy adult)", "My child", "My patient", "Family member"])
with col_b:
    gender = st.selectbox("Gender", ["Female", "Male"])
with col_c:
    age_group = st.selectbox("Age group", ["Child (2–12 yrs)", "Teen (13–17 yrs)", "Adult (18–59 yrs)", "Senior (60+ yrs)"])

col_d, col_e = st.columns(2)
with col_d:
    activity = st.selectbox("Activity level", ["Sedentary (little/no exercise)", "Light (1–3 days/week)", "Moderate (3–5 days/week)", "Active (6–7 days/week)"])
with col_e:
    goal = st.selectbox("Health goal", ["Maintain healthy weight", "Lose weight", "Gain weight / build strength", "Manage illness / recovery"])

# ── Conditions ───────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">🏥 Select Health Condition(s)</div>', unsafe_allow_html=True)
st.markdown("Select one or more conditions. A combined safe meal plan will be generated.")

selected_conditions = st.multiselect(
    "Health conditions (leave empty for general healthy diet)",
    list(CONDITIONS.keys()),
    placeholder="Choose conditions..."
)

# ── Allergies / Preferences ──────────────────────────────────────────────────
st.markdown('<div class="section-title">🚫 Dietary Preferences / Allergies</div>', unsafe_allow_html=True)
col_p1, col_p2, col_p3, col_p4 = st.columns(4)
with col_p1: no_gluten  = st.checkbox("Gluten-free")
with col_p2: no_dairy   = st.checkbox("Dairy-free")
with col_p3: vegetarian = st.checkbox("Vegetarian")
with col_p4: no_seafood = st.checkbox("No seafood")

# ── Generate button ───────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
generate = st.button("🥗 GENERATE MY MEAL PLAN")

# ── Output ───────────────────────────────────────────────────────────────────
if generate:

    # ── Condition badges ─────────────────────────────────────────────────────
    if selected_conditions:
        badges = ""
        for c in selected_conditions:
            color, bg = CONDITION_COLORS[c]
            badges += f'<span class="condition-badge" style="background:{bg};color:{color};border:1.5px solid {color};">{CONDITIONS[c]["emoji"]} {c}</span>'
        st.markdown(f'<div style="margin-bottom:8px;">{badges}</div>', unsafe_allow_html=True)

        # Combined key messages
        for c in selected_conditions:
            st.markdown(f"""
            <div class="info-box">
              <strong>{CONDITIONS[c]["emoji"]} {c}:</strong> {CONDITIONS[c]["description"]}
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
          <strong>🌿 General Healthy Diet</strong> — Balanced nutrition for a healthy active lifestyle.
          Includes whole grains, lean protein, healthy fats, and plenty of vegetables.
        </div>""", unsafe_allow_html=True)

    # ── Calorie adjustment ───────────────────────────────────────────────────
    activity_mult = {"Sedentary (little/no exercise)": 1.0, "Light (1–3 days/week)": 1.1,
                     "Moderate (3–5 days/week)": 1.2,       "Active (6–7 days/week)": 1.35}
    goal_mult     = {"Maintain healthy weight": 1.0, "Lose weight": 0.85,
                     "Gain weight / build strength": 1.15,   "Manage illness / recovery": 1.0}

    if selected_conditions:
        base_cal = sum(CONDITIONS[c]["calories"][0] + CONDITIONS[c]["calories"][1]
                       for c in selected_conditions) / (2 * len(selected_conditions))
    else:
        base_cal = 1800

    target_cal = int(base_cal * activity_mult[activity] * goal_mult[goal])

    st.markdown(f"""
    <div class="meal-card" style="border-color:#10b981;background:#f0fdf4;">
      <h4>🎯 Your Daily Calorie Target: ~{target_cal} kcal</h4>
      <div style="color:#065f46;font-size:0.9rem;">
        Based on: {user_type} · {gender} · {age_group} · {activity} · Goal: {goal}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Meal plan ────────────────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🍽️ Your Daily Meal Plan</div>', unsafe_allow_html=True)

    plan_key = selected_conditions[0] if len(selected_conditions) == 1 else (
        selected_conditions[0] if selected_conditions else "general"
    )
    plan = MEAL_PLANS.get(plan_key, MEAL_PLANS["general"])

    total_cal = total_carb = total_protein = total_fat = 0

    for slot, (emoji, label) in MEAL_LABELS.items():
        meal = plan[slot]

        # Apply dietary preferences
        items = meal["items"].copy()
        if no_dairy:
            items = [i for i in items if not any(w in i.lower() for w in ["milk","yogurt","cheese","dairy","butter"])]
            if not items:
                items = ["Dairy-free alternative (oat milk, soy milk, coconut yogurt)"]
        if no_seafood:
            items = [i for i in items if not any(w in i.lower() for w in ["fish","tuna","salmon","sardine","mackerel","seafood","prawn"])]
            if not items:
                items = ["Chicken breast or tofu (seafood-free substitute)"]
        if vegetarian:
            items = [i for i in items if not any(w in i.lower() for w in ["chicken","beef","meat","fish","tuna","salmon","sardine","mackerel"])]
            if not items:
                items = ["Tofu, tempeh, or legumes (vegetarian substitute)"]

        items_html = "".join(f'<div class="meal-item">• {i}</div>' for i in items)
        nutrient_html = f"""
        <div class="nutrient-row">
          <span class="nutrient-chip">🔥 {meal['cal']} kcal</span>
          <span class="nutrient-chip">🍞 Carbs: {meal['carb']}g</span>
          <span class="nutrient-chip">🥩 Protein: {meal['protein']}g</span>
          <span class="nutrient-chip">🫒 Fat: {meal['fat']}g</span>
        </div>"""

        st.markdown(f"""
        <div class="meal-card">
          <h4>{emoji} {label} — {meal['name']}</h4>
          {items_html}
          {nutrient_html}
        </div>""", unsafe_allow_html=True)

        total_cal     += meal["cal"]
        total_carb    += meal["carb"]
        total_protein += meal["protein"]
        total_fat     += meal["fat"]

    # ── Daily total ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="meal-card" style="border-color:#059669;background:#ecfdf5;">
      <h4>📊 Daily Nutrition Total</h4>
      <div class="nutrient-row">
        <span class="nutrient-chip" style="background:#d1fae5;border-color:#059669;">🔥 Total: {total_cal} kcal</span>
        <span class="nutrient-chip" style="background:#d1fae5;border-color:#059669;">🍞 Carbs: {total_carb}g</span>
        <span class="nutrient-chip" style="background:#d1fae5;border-color:#059669;">🥩 Protein: {total_protein}g</span>
        <span class="nutrient-chip" style="background:#d1fae5;border-color:#059669;">🫒 Fat: {total_fat}g</span>
      </div>
      <div style="margin-top:8px;color:#065f46;font-size:0.88rem;">
        Target: ~{target_cal} kcal/day &nbsp;|&nbsp;
        Difference: {total_cal - target_cal:+d} kcal
        {"✅ On target!" if abs(total_cal - target_cal) < 150 else ("⬆️ Slightly above" if total_cal > target_cal else "⬇️ Slightly below")}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Safe & Avoid Foods ───────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">✅ Safe Foods & ❌ Foods to Avoid</div>', unsafe_allow_html=True)

    if selected_conditions:
        for cond in selected_conditions:
            color, bg = CONDITION_COLORS[cond]
            info = CONDITIONS[cond]

            safe_items  = "".join(f'<div class="safe-food">✅ {f}</div>' for f in info["safe"])
            avoid_items = "".join(f'<div class="avoid-food">❌ {f}</div>' for f in info["avoid"])

            col_s, col_a = st.columns(2)
            with col_s:
                st.markdown(f"""
                <div class="meal-card" style="border-color:{color};background:{bg};">
                  <h4 style="color:{color};">{info['emoji']} {cond} — SAFE Foods</h4>
                  {safe_items}
                </div>""", unsafe_allow_html=True)
            with col_a:
                st.markdown(f"""
                <div class="meal-card" style="border-color:#dc2626;background:#fff5f5;">
                  <h4 style="color:#dc2626;">⚠️ {cond} — AVOID These Foods</h4>
                  {avoid_items}
                </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="meal-card">
          <h4>🌿 General Healthy Eating — Key Tips</h4>
          <div class="safe-food">✅ Eat plenty of vegetables and fruits (5 portions/day)</div>
          <div class="safe-food">✅ Choose whole grains over refined grains</div>
          <div class="safe-food">✅ Include lean protein at every meal</div>
          <div class="safe-food">✅ Use healthy fats (olive oil, nuts, avocado)</div>
          <div class="safe-food">✅ Drink 8 glasses of water daily</div>
          <div class="avoid-food">❌ Limit ultra-processed foods and fast food</div>
          <div class="avoid-food">❌ Reduce added sugar and sugary drinks</div>
          <div class="avoid-food">❌ Limit salt and high-sodium condiments</div>
          <div class="avoid-food">❌ Avoid trans fats (margarine, fried snacks)</div>
        </div>""", unsafe_allow_html=True)

    # ── Disclaimer ───────────────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="warning-box">
      <strong>⚠️ Medical Disclaimer</strong><br>
      This meal plan is for general educational guidance only and is NOT a substitute for professional medical or
      dietetic advice. For patients with critical illness, please consult a registered dietitian or doctor before
      making dietary changes. Individual nutritional needs vary based on medications, lab results, and disease stage.
    </div>
    """, unsafe_allow_html=True)
