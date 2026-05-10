import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FitGenie AI",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0e1117; }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #e94560;
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 0 40px rgba(233,69,96,0.15);
    }
    .hero-header h1 {
        color: #fff;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
    }
    .hero-header p {
        color: #adb5bd;
        font-size: 1.1rem;
        margin: 0.5rem 0 0;
    }
    .hero-accent { color: #e94560; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e1e2e, #252535);
        border: 1px solid #2d2d45;
        border-radius: 16px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(233,69,96,0.12);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #e94560;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #6c757d;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #adb5bd;
        margin-top: 0.2rem;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 2px solid #e94560;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    .result-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e94560;
        margin-bottom: 0.8rem;
    }

    /* Day plan rows */
    .day-row {
        display: flex;
        align-items: flex-start;
        padding: 0.55rem 0;
        border-bottom: 1px solid #2d2d45;
    }
    .day-name {
        min-width: 100px;
        font-weight: 600;
        color: #e94560;
        font-size: 0.85rem;
    }
    .day-workout { color: #dee2e6; font-size: 0.9rem; }

    /* Adjustment item */
    .adj-item {
        background: #1e1e2e;
        border-left: 3px solid #e94560;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 1rem;
        margin: 0.3rem 0;
        color: #dee2e6;
        font-size: 0.9rem;
    }

    /* Section header */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #fff;
        border-left: 4px solid #e94560;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #12121e;
        border-right: 1px solid #2d2d45;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #adb5bd !important;
        font-size: 0.85rem;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c62a47);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 2.5rem;
        font-weight: 700;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s;
        letter-spacing: 0.02em;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #ff5c7a, #e94560);
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(233,69,96,0.35);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #12121e;
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #6c757d;
        border-radius: 8px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #e94560 !important;
        color: white !important;
    }

    /* Progress bar override */
    .stProgress .st-bo { background: #e94560; }

    /* Score circle */
    .score-circle {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: conic-gradient(#e94560 var(--pct), #2d2d45 0);
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        position: relative;
    }
    .score-inner {
        width: 90px;
        height: 90px;
        border-radius: 50%;
        background: #12121e;
        display: flex;
        align-items: center;
        justify-content: center;
        position: absolute;
    }
    .score-text {
        font-size: 1.4rem;
        font-weight: 800;
        color: #e94560;
        z-index: 1;
    }

    /* Info box */
    .info-box {
        background: #1a2744;
        border: 1px solid #1a4fa0;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        color: #90b4e8;
        font-size: 0.88rem;
        margin: 0.5rem 0;
    }
    .warn-box {
        background: #2a1f00;
        border: 1px solid #a07000;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        color: #e8c842;
        font-size: 0.88rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

WORKOUT_PLANS = {
    "Fat Loss": {
        "Weekly Plan": {
            "Monday":    "HIIT — 30 min (Burpees, Jump Squats, Mountain Climbers)",
            "Tuesday":   "Cardio — 45 min (Treadmill / Cycling)",
            "Wednesday": "Strength — Full Body (Squats, Deadlifts, Bench Press)",
            "Thursday":  "HIIT — 30 min (Tabata Protocol)",
            "Friday":    "Cardio — 40 min (Elliptical / Rowing)",
            "Saturday":  "Strength — Upper Body Focus",
            "Sunday":    "Rest / Active Recovery (Light Walk / Stretching)",
        },
        "Diet Tip": "Caloric deficit of 300–500 kcal/day. High protein (1.8g/kg body weight). Low GI carbs.",
        "Intensity": "Moderate-High",
        "Primary Goal": "Reduce body fat, improve cardiovascular fitness",
        "color": "#ff6b6b",
        "icon": "🔥",
    },
    "Muscle Gain": {
        "Weekly Plan": {
            "Monday":    "Push — Chest, Shoulders, Triceps (Bench Press, OHP, Dips)",
            "Tuesday":   "Pull — Back, Biceps (Deadlifts, Pull-Ups, Rows)",
            "Wednesday": "Legs — Quads, Hamstrings, Glutes (Squats, Leg Press, RDL)",
            "Thursday":  "Push — Heavy (Progressive Overload +2.5kg each week)",
            "Friday":    "Pull — Heavy (Progressive Overload)",
            "Saturday":  "Legs — Accessory & Core Work",
            "Sunday":    "Rest / Foam Rolling",
        },
        "Diet Tip": "Caloric surplus of 200–300 kcal/day. Protein 2.2g/kg. Carbs pre/post workout.",
        "Intensity": "High (Progressive Overload)",
        "Primary Goal": "Hypertrophy and strength gains",
        "color": "#51cf66",
        "icon": "💪",
    },
    "Endurance / Maintenance": {
        "Weekly Plan": {
            "Monday":    "Long Run — 60 min (Zone 2 Cardio, 60–70% Max HR)",
            "Tuesday":   "Strength — Full Body Circuit",
            "Wednesday": "Cross-Training — Swimming / Cycling 45 min",
            "Thursday":  "Tempo Run — 30 min (80% Max HR)",
            "Friday":    "Yoga / Flexibility & Core",
            "Saturday":  "Long Cardio — 75 min (VO₂ Max Training)",
            "Sunday":    "Rest / Light Walk",
        },
        "Diet Tip": "Balanced macros. Complex carbs pre-workout. Electrolyte replenishment post-cardio.",
        "Intensity": "Moderate (Volume-focused)",
        "Primary Goal": "Build aerobic base and maintain fitness",
        "color": "#339af0",
        "icon": "🏃",
    },
    "Strength / Advanced": {
        "Weekly Plan": {
            "Monday":    "Powerlifting — Squat Focus (5×5 Heavy)",
            "Tuesday":   "Olympic Lifting — Clean & Jerk, Snatch",
            "Wednesday": "Accessory Work — Isolation & Weak Points",
            "Thursday":  "Powerlifting — Bench Focus (5×5 Heavy)",
            "Friday":    "Powerlifting — Deadlift Focus",
            "Saturday":  "Conditioning — Sled Push, Farmer Carry, Battle Ropes",
            "Sunday":    "Active Recovery — Mobility Drills",
        },
        "Diet Tip": "High calorie intake (TDEE +400 kcal). Protein 2.5g/kg. Creatine supplementation beneficial.",
        "Intensity": "Very High (Max Strength)",
        "Primary Goal": "Maximise strength and power output",
        "color": "#cc5de8",
        "icon": "🏆",
    },
}

CLUSTER_FEATURES = [
    "BMI", "Fat_Percentage", "Experience_Level",
    "Workout_Frequency (days/week)", "Calories_Burned",
    "Session_Duration (hours)", "Avg_BPM",
]

REG_FEATURES = [
    "Age", "Weight (kg)", "BMI", "Fat_Percentage",
    "Avg_BPM", "Max_BPM", "Resting_BPM",
    "Session_Duration (hours)", "Workout_Frequency (days/week)",
    "Experience_Level", "Gender_Encoded", "Workout_Type_Encoded",
]

CLF_FEATURES = [
    "Age", "Weight (kg)", "BMI", "Fat_Percentage",
    "Avg_BPM", "Session_Duration (hours)",
    "Workout_Frequency (days/week)", "Experience_Level", "Gender_Encoded",
]


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25.0:
        return "Normal"
    elif bmi < 30.0:
        return "Overweight"
    else:
        return "Obese"


def label_cluster(row):
    if row["Avg_BMI"] > 28 and row["Avg_Experience"] <= 1.5:
        return "Fat Loss"
    elif row["Avg_Experience"] >= 2.5:
        return "Strength / Advanced"
    elif row["Avg_Frequency"] >= 4 and row["Avg_BMI"] < 26:
        return "Muscle Gain"
    else:
        return "Endurance / Maintenance"


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATASET (no CSV needed)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def generate_dataset(n=900):
    np.random.seed(42)
    genders    = np.random.choice(["Male", "Female"], n)
    ages       = np.random.randint(18, 65, n)
    heights    = np.where(genders == "Male",
                          np.random.normal(1.76, 0.07, n),
                          np.random.normal(1.63, 0.07, n)).clip(1.5, 2.1)
    weights    = np.where(genders == "Male",
                          np.random.normal(80, 14, n),
                          np.random.normal(65, 12, n)).clip(40, 140)
    bmis       = weights / heights**2
    fat_pct    = (bmis * 1.2 - 5 +
                  np.where(genders == "Female", 8, 0) +
                  np.random.normal(0, 3, n)).clip(5, 45)
    exp        = np.random.choice([1, 2, 3], n, p=[0.45, 0.35, 0.20])
    freq       = np.random.choice([2, 3, 4, 5, 6], n, p=[0.15,0.30,0.30,0.15,0.10])
    duration   = (0.5 + exp * 0.2 + np.random.normal(0, 0.2, n)).clip(0.5, 3.0)
    rest_bpm   = np.random.randint(50, 85, n).astype(float)
    max_bpm    = (220 - ages + np.random.normal(0, 5, n)).clip(150, 210)
    avg_bpm    = (rest_bpm + (max_bpm - rest_bpm) * np.random.uniform(0.55, 0.85, n)).clip(90, 195)
    wtype      = np.random.choice(["Cardio", "HIIT", "Strength", "Yoga"], n)
    calories   = (
        duration * avg_bpm * 0.8
        + weights * 0.3
        + exp * 50
        + np.random.normal(0, 60, n)
    ).clip(200, 1500)
    water      = (duration * 0.5 + np.random.normal(0, 0.3, n)).clip(1, 4)

    df = pd.DataFrame({
        "Age": ages,
        "Gender": genders,
        "Weight (kg)": weights.round(1),
        "Height (m)": heights.round(2),
        "BMI": bmis.round(2),
        "Fat_Percentage": fat_pct.round(1),
        "Experience_Level": exp,
        "Workout_Frequency (days/week)": freq,
        "Session_Duration (hours)": duration.round(2),
        "Resting_BPM": rest_bpm.astype(int),
        "Max_BPM": max_bpm.astype(int),
        "Avg_BPM": avg_bpm.round(0).astype(int),
        "Workout_Type": wtype,
        "Calories_Burned": calories.round(0).astype(int),
        "Water_Intake (liters)": water.round(2),
    })
    return df


@st.cache_resource(show_spinner=False)
def train_models():
    df = generate_dataset()

    le_gender  = LabelEncoder()
    le_workout = LabelEncoder()
    df["Gender_Encoded"]       = le_gender.fit_transform(df["Gender"])
    df["Workout_Type_Encoded"] = le_workout.fit_transform(df["Workout_Type"])

    # Clustering
    cluster_scaler = StandardScaler()
    X_cl = cluster_scaler.fit_transform(df[CLUSTER_FEATURES])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_cl)

    profile = df.groupby("Cluster").agg(
        Avg_BMI=("BMI", "mean"),
        Avg_Fat_Pct=("Fat_Percentage", "mean"),
        Avg_Experience=("Experience_Level", "mean"),
        Avg_Calories=("Calories_Burned", "mean"),
        Avg_Frequency=("Workout_Frequency (days/week)", "mean"),
    ).round(2)
    profile["Fitness_Goal"] = profile.apply(label_cluster, axis=1)
    cluster_map = profile["Fitness_Goal"].to_dict()
    df["Fitness_Goal"] = df["Cluster"].map(cluster_map)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_cl)

    # Regression
    X_reg = df[REG_FEATURES]
    y_reg = df["Calories_Burned"]
    X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    reg_scaler = StandardScaler()
    X_tr_sc = reg_scaler.fit_transform(X_tr)
    X_te_sc  = reg_scaler.transform(X_te)

    lr  = LinearRegression().fit(X_tr_sc, y_tr)
    rf  = RandomForestRegressor(n_estimators=150, random_state=42).fit(X_tr, y_tr)

    models_info = {}
    for name, m, Xtr_, Xte_ in [
        ("Linear Regression", lr, X_tr_sc, X_te_sc),
        ("Random Forest",     rf, X_tr,    X_te),
    ]:
        p = m.predict(Xte_)
        models_info[name] = {
            "MAE":  round(mean_absolute_error(y_te, p), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_te, p)), 2),
            "R2":   round(r2_score(y_te, p), 4),
        }

    # Classification
    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(df[CLF_FEATURES], df["Workout_Type_Encoded"])

    return {
        "df": df,
        "le_gender": le_gender,
        "le_workout": le_workout,
        "cluster_scaler": cluster_scaler,
        "kmeans": kmeans,
        "cluster_map": cluster_map,
        "pca": pca,
        "X_pca": X_pca,
        "reg_scaler": reg_scaler,
        "lr": lr,
        "rf": rf,
        "clf": clf,
        "models_info": models_info,
        "reg_features": REG_FEATURES,
        "X_te": X_te,
        "X_te_sc": X_te_sc,
        "y_te": y_te,
        "profile": profile,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def predict_for_user(user, state):
    bmi = user["Weight_kg"] / user["Height_m"] ** 2
    fv = np.array([[
        bmi, user["Fat_Percentage"], user["Experience_Level"],
        user["Workout_Frequency"], user.get("Calories_Burned", 700),
        user["Session_Duration_hrs"], user["Avg_BPM"],
    ]])
    fv_sc  = state["cluster_scaler"].transform(fv)
    cluster = state["kmeans"].predict(fv_sc)[0]
    goal    = state["cluster_map"][int(cluster)]

    gender_enc = 1 if user["Gender"] == "Male" else 0
    reg_in = pd.DataFrame([{
        "Age": user["Age"],
        "Weight (kg)": user["Weight_kg"],
        "BMI": bmi,
        "Fat_Percentage": user["Fat_Percentage"],
        "Avg_BPM": user["Avg_BPM"],
        "Max_BPM": user["Avg_BPM"] + 20,
        "Resting_BPM": user["Resting_BPM"],
        "Session_Duration (hours)": user["Session_Duration_hrs"],
        "Workout_Frequency (days/week)": user["Workout_Frequency"],
        "Experience_Level": user["Experience_Level"],
        "Gender_Encoded": gender_enc,
        "Workout_Type_Encoded": 0,
    }])
    cal_pred = state["rf"].predict(reg_in)[0]

    clf_in = pd.DataFrame([{
        "Age": user["Age"],
        "Weight (kg)": user["Weight_kg"],
        "BMI": bmi,
        "Fat_Percentage": user["Fat_Percentage"],
        "Avg_BPM": user["Avg_BPM"],
        "Session_Duration (hours)": user["Session_Duration_hrs"],
        "Workout_Frequency (days/week)": user["Workout_Frequency"],
        "Experience_Level": user["Experience_Level"],
        "Gender_Encoded": gender_enc,
    }])
    wtype_enc = state["clf"].predict(clf_in)[0]
    wtype     = state["le_workout"].inverse_transform([wtype_enc])[0]

    return {
        "BMI": round(bmi, 2),
        "BMI_Category": bmi_category(bmi),
        "Cluster": int(cluster),
        "Fitness_Goal": goal,
        "Calories_Predicted": round(cal_pred),
        "Workout_Type": wtype,
        **WORKOUT_PLANS[goal],
    }


def adaptive_feedback(plan, feedback):
    difficulty    = feedback["perceived_difficulty"]
    sessions_done = feedback["sessions_completed"]
    weight_change = feedback["weight_change_kg"]
    goal          = plan["Fitness_Goal"]
    adjustments   = []

    if difficulty == 1:
        adjustments += [
            "⬆️  Increase weight/resistance by 5–10%",
            "⬆️  Add 1 extra set to compound exercises",
            "⬆️  Reduce rest periods by 15 seconds",
        ]
    elif difficulty == 3:
        adjustments += [
            "⬇️  Reduce weight/resistance by 10%",
            "⬇️  Drop 1 set from main exercises",
            "⬇️  Increase rest periods by 30 seconds",
            "⬇️  Replace 1 HIIT session with steady-state cardio",
        ]
    else:
        adjustments += [
            "✅  Intensity is appropriate — maintain current level",
            "✅  Consider progressive overload next week (+2.5 kg)",
        ]

    if sessions_done < 4:
        adjustments += [
            "📅  Schedule workouts at fixed times to improve adherence",
            "📅  Consider reducing to a 4-day plan for sustainability",
        ]

    if goal == "Fat Loss" and weight_change > 0:
        adjustments.append("🍽️  Review diet — caloric surplus detected; reduce carbs by 50 g/day")
    elif goal == "Muscle Gain" and weight_change < 0:
        adjustments.append("🍽️  Increase caloric intake by 200 kcal/day; add post-workout protein shake")

    perf = min(100, int(
        (sessions_done / 7) * 50 +
        (3 - abs(difficulty - 2)) * 20 +
        (1 if (weight_change <= 0 and goal == "Fat Loss") or
              (weight_change >= 0 and goal == "Muscle Gain") else 0.5) * 30
    ))
    return {"Performance_Score": perf, "Adjustments": adjustments,
            "Next_Week_Goal": f"Maintain {goal} focus with adjusted intensity"}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Hero
    st.markdown("""
    <div class="hero-header">
        <h1>🏋️ FitGenie <span class="hero-accent">AI</span></h1>
        <p>Personalized Fitness Recommendation System &nbsp;|&nbsp; K-Means Clustering · Regression · Classification</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🔧 Training models on synthetic gym dataset…"):
        state = train_models()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 👤 Your Profile")
        st.markdown("---")

        gender = st.selectbox("Gender", ["Male", "Female"])
        age    = st.slider("Age", 16, 75, 28)
        col1, col2 = st.columns(2)
        with col1:
            weight = st.number_input("Weight (kg)", 40.0, 160.0, 80.0, 0.5)
        with col2:
            height = st.number_input("Height (m)", 1.40, 2.20, 1.75, 0.01)

        st.markdown("---")
        st.markdown("**💓 Heart Rate**")
        avg_bpm     = st.slider("Avg BPM during workout", 80, 200, 145)
        resting_bpm = st.slider("Resting BPM", 40, 100, 65)

        st.markdown("---")
        st.markdown("**🏃 Workout Details**")
        experience = st.selectbox("Experience Level", [1, 2, 3],
                                   format_func=lambda x: {1:"Beginner",2:"Intermediate",3:"Advanced"}[x])
        frequency   = st.slider("Workout Frequency (days/week)", 1, 7, 3)
        duration    = st.slider("Session Duration (hours)", 0.25, 3.0, 1.0, 0.25)
        fat_pct     = st.slider("Body Fat %", 5.0, 50.0, 20.0, 0.5)
        calories_known = st.number_input("Approx. Calories Burned/session (0 = estimate)", 0, 2000, 0, 50)

        st.markdown("---")
        predict_btn = st.button("🚀 Generate My Plan")

    user_data = {
        "Age": age, "Gender": gender,
        "Weight_kg": weight, "Height_m": height,
        "Avg_BPM": avg_bpm, "Resting_BPM": resting_bpm,
        "Session_Duration_hrs": duration,
        "Fat_Percentage": fat_pct,
        "Workout_Frequency": frequency,
        "Experience_Level": experience,
        "Calories_Burned": calories_known if calories_known > 0 else int(state["df"]["Calories_Burned"].mean()),
    }

    if "result" not in st.session_state:
        st.session_state["result"] = None
    if predict_btn:
        st.session_state["result"] = predict_for_user(user_data, state)

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "🎯 My Plan",
        "📊 Data Insights",
        "🔵 Clustering",
        "🔥 Calorie Predictor",
        "🔄 Adaptive Feedback",
        "📈 Progress Dashboard",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — MY PLAN
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        if st.session_state["result"] is None:
            st.markdown("""
            <div class="info-box">
                👈  Fill in your profile in the sidebar and click <strong>Generate My Plan</strong> to get started.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### How FitGenie AI works")
            c1, c2, c3, c4 = st.columns(4)
            steps = [
                ("1️⃣", "Profile Input", "Enter age, weight, BPM, experience & more"),
                ("2️⃣", "K-Means Clustering", "You're matched to a fitness cluster"),
                ("3️⃣", "ML Prediction", "Calories & workout type predicted"),
                ("4️⃣", "Full Plan", "7-day plan + diet tips + adaptive feedback"),
            ]
            for col, (icon, title, desc) in zip([c1,c2,c3,c4], steps):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:2rem">{icon}</div>
                        <div class="metric-label" style="font-size:0.9rem;color:#fff;margin-top:0.4rem">{title}</div>
                        <div class="metric-sub">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)
            return

        r   = st.session_state["result"]
        goal = r["Fitness_Goal"]
        plan_info = WORKOUT_PLANS[goal]
        accent = plan_info["color"]

        # ── Metrics row
        bmi_val = r["BMI"]
        bmi_cat = r["BMI_Category"]
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, val, label, sub in [
            (c1, f"{bmi_val}", "BMI", bmi_cat),
            (c2, f"{fat_pct}%", "Body Fat", "Measured"),
            (c3, f"{r['Calories_Predicted']}", "Est. Cal Burn", "kcal/session"),
            (c4, r["Workout_Type"], "Suggested Type", "Classifier"),
            (c5, plan_info["Intensity"], "Intensity", "Level"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{accent}">{val}</div>
                    <div class="metric-label">{label}</div>
                    <div class="metric-sub">{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Goal banner
        st.markdown(f"""
        <div class="result-card" style="border-color:{accent}; text-align:center">
            <div style="font-size:3rem">{plan_info['icon']}</div>
            <div style="font-size:1.8rem; font-weight:800; color:{accent}">{goal}</div>
            <div style="color:#adb5bd; margin-top:0.4rem">{plan_info['Primary Goal']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_plan, col_diet = st.columns([3, 2])

        with col_plan:
            st.markdown('<div class="section-header">📅 7-Day Workout Plan</div>', unsafe_allow_html=True)
            day_colors = {
                "Monday":"#ff6b6b","Tuesday":"#ffa94d","Wednesday":"#51cf66",
                "Thursday":"#339af0","Friday":"#cc5de8","Saturday":"#f06595","Sunday":"#74c0fc",
            }
            for day, workout in plan_info["Weekly Plan"].items():
                st.markdown(f"""
                <div class="day-row">
                    <span class="day-name" style="color:{day_colors.get(day,'#e94560')}">{day}</span>
                    <span class="day-workout">{workout}</span>
                </div>
                """, unsafe_allow_html=True)

        with col_diet:
            st.markdown('<div class="section-header">🥗 Nutrition Tip</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="result-card" style="border-color:{accent}">
                <div style="color:#dee2e6;line-height:1.7">{plan_info['Diet Tip']}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-header">📊 Your Stats</div>', unsafe_allow_html=True)
            max_bpm_est = 220 - age
            water_rec   = round(duration * 0.5 + weight * 0.033, 1)

            for label, val in [
                ("🫀 Max BPM (estimated)", f"{max_bpm_est} bpm"),
                ("💧 Water Intake Recommendation", f"{water_rec} L/day"),
                ("🔥 Workout Intensity", plan_info["Intensity"]),
                ("📅 Frequency", f"{frequency} days/week"),
                ("⏱️ Session Duration", f"{duration} hrs"),
            ]:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:0.4rem 0;
                            border-bottom:1px solid #2d2d45;color:#dee2e6;font-size:0.9rem">
                    <span>{label}</span>
                    <span style="color:{accent};font-weight:600">{val}</span>
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — DATA INSIGHTS
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        df = state["df"]
        st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        for col, val, label in [
            (c1, len(df), "Total Users"),
            (c2, f"{df['BMI'].mean():.1f}", "Avg BMI"),
            (c3, f"{df['Calories_Burned'].mean():.0f}", "Avg Cal Burn"),
            (c4, f"{df['Fat_Percentage'].mean():.1f}%", "Avg Body Fat"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            fig = make_subplots(rows=2, cols=3,
                subplot_titles=["Age","BMI","Weight (kg)","Calories Burned","Fat %","Session Duration"])
            cols_p = ["Age","BMI","Weight (kg)","Calories_Burned","Fat_Percentage","Session_Duration (hours)"]
            positions = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
            colors_p  = ["#e94560","#51cf66","#339af0","#ffa94d","#cc5de8","#74c0fc"]
            for c, pos, clr in zip(cols_p, positions, colors_p):
                fig.add_trace(go.Histogram(x=df[c], showlegend=False,
                                           marker_color=clr, opacity=0.8), row=pos[0], col=pos[1])
            fig.update_layout(height=480, template="plotly_dark",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              title="Distribution of Key Features", title_font_color="#fff")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = px.pie(df, names="Workout_Type", title="Workout Type Distribution",
                          color_discrete_sequence=["#e94560","#51cf66","#339af0","#cc5de8"])
            fig2.update_layout(template="plotly_dark",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               title_font_color="#fff")
            st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)
        exp_map = {1:"Beginner",2:"Intermediate",3:"Advanced"}
        df["Experience_Label"] = df["Experience_Level"].map(exp_map)
        with c3:
            fig3 = px.box(df, x="Workout_Type", y="Calories_Burned",
                          color="Experience_Label",
                          title="Calories Burned by Workout Type & Experience",
                          color_discrete_sequence=["#339af0","#51cf66","#e94560"])
            fig3.update_layout(template="plotly_dark",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               title_font_color="#fff")
            st.plotly_chart(fig3, use_container_width=True)

        with c4:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            corr = df[num_cols].corr()
            fig4 = px.imshow(corr, text_auto=".2f", aspect="auto",
                             title="Feature Correlation Heatmap",
                             color_continuous_scale="RdBu_r")
            fig4.update_layout(height=420, template="plotly_dark",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               title_font_color="#fff")
            st.plotly_chart(fig4, use_container_width=True)

        with st.expander("🗂️ View Raw Dataset Sample"):
            st.dataframe(df.head(50), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — CLUSTERING
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        df  = state["df"]
        st.markdown('<div class="section-header">🔵 K-Means Clustering Analysis</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            # Elbow / Silhouette
            X_cl = state["cluster_scaler"].transform(df[CLUSTER_FEATURES])
            inertias, sils = [], []
            K_range = range(2, 9)
            for k in K_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X_cl)
                inertias.append(km.inertia_)
                from sklearn.metrics import silhouette_score as sil_score
                sils.append(sil_score(X_cl, km.labels_))

            fig_elbow = make_subplots(rows=1, cols=2, subplot_titles=["Elbow (Inertia)", "Silhouette Score"])
            fig_elbow.add_trace(go.Scatter(x=list(K_range), y=inertias, mode="lines+markers",
                                           marker_color="#e94560", name="Inertia"), row=1, col=1)
            fig_elbow.add_trace(go.Scatter(x=list(K_range), y=sils, mode="lines+markers",
                                           marker_color="#51cf66", name="Silhouette"), row=1, col=2)
            fig_elbow.update_layout(template="plotly_dark",
                                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    title="Optimal K Selection", title_font_color="#fff", height=360)
            st.plotly_chart(fig_elbow, use_container_width=True)

        with c2:
            # PCA scatter
            X_pca = state["X_pca"]
            fig_pca = px.scatter(x=X_pca[:,0], y=X_pca[:,1],
                                 color=df["Fitness_Goal"],
                                 title="User Clusters — PCA 2D Projection",
                                 labels={"x":"PCA 1","y":"PCA 2","color":"Fitness Goal"},
                                 color_discrete_sequence=["#e94560","#51cf66","#339af0","#cc5de8"])
            fig_pca.update_layout(template="plotly_dark",
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  title_font_color="#fff", height=360)
            st.plotly_chart(fig_pca, use_container_width=True)

        # Cluster profiles table
        st.markdown('<div class="section-header">📋 Cluster Profiles</div>', unsafe_allow_html=True)
        profile = state["profile"].reset_index()
        st.dataframe(profile.style.background_gradient(cmap="RdYlGn", subset=["Avg_BMI","Avg_Fat_Pct"]),
                     use_container_width=True)

        # Goal distribution
        st.markdown('<div class="section-header">📊 Fitness Goal Distribution</div>', unsafe_allow_html=True)
        goal_counts = df["Fitness_Goal"].value_counts().reset_index()
        goal_counts.columns = ["Goal","Count"]
        fig_bar = px.bar(goal_counts, x="Goal", y="Count", color="Goal",
                         color_discrete_sequence=["#e94560","#51cf66","#339af0","#cc5de8"])
        fig_bar.update_layout(template="plotly_dark",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              showlegend=False, title_font_color="#fff")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — CALORIE PREDICTOR
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown('<div class="section-header">🔥 Calorie Burn Prediction</div>', unsafe_allow_html=True)
        models_info = state["models_info"]

        c1, c2, c3 = st.columns(3)
        for col, (name, info) in zip([c1,c2,c3], models_info.items()):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:0.95rem;font-weight:700;color:#fff;margin-bottom:0.5rem">{name}</div>
                    <div class="metric-value">{info['R2']}</div>
                    <div class="metric-label">R² Score</div>
                    <div class="metric-sub">MAE: {info['MAE']} | RMSE: {info['RMSE']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            # Actual vs Predicted
            y_te  = state["y_te"]
            X_te  = state["X_te"]
            preds = state["rf"].predict(X_te)
            fig_av = px.scatter(x=y_te, y=preds,
                                labels={"x":"Actual Calories","y":"Predicted Calories"},
                                title="Actual vs Predicted — Random Forest",
                                color_discrete_sequence=["#e94560"])
            fig_av.add_shape(type="line",
                             x0=y_te.min(), y0=y_te.min(),
                             x1=y_te.max(), y1=y_te.max(),
                             line=dict(color="#51cf66", dash="dash"))
            fig_av.update_layout(template="plotly_dark",
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 title_font_color="#fff")
            st.plotly_chart(fig_av, use_container_width=True)

        with c2:
            # Feature importance
            fi = pd.Series(state["rf"].feature_importances_, index=REG_FEATURES).sort_values()
            fig_fi = px.bar(x=fi.values, y=fi.index, orientation="h",
                            title="Feature Importance — Random Forest",
                            labels={"x":"Importance","y":""},
                            color=fi.values, color_continuous_scale="OrRd")
            fig_fi.update_layout(template="plotly_dark",
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 title_font_color="#fff", coloraxis_showscale=False)
            st.plotly_chart(fig_fi, use_container_width=True)

        # ── Interactive predictor
        st.markdown('<div class="section-header">🎛️ Try the Calorie Predictor</div>', unsafe_allow_html=True)
        with st.form("cal_form"):
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                p_age    = st.number_input("Age", 16, 75, int(age))
                p_weight = st.number_input("Weight (kg)", 40.0, 160.0, float(weight))
                p_bmi    = p_weight / (height**2)
            with cc2:
                p_bpm     = st.number_input("Avg BPM", 80, 200, int(avg_bpm))
                p_restbpm = st.number_input("Resting BPM", 40, 100, int(resting_bpm))
                p_dur     = st.number_input("Duration (hrs)", 0.25, 3.0, float(duration))
            with cc3:
                p_fat   = st.number_input("Body Fat %", 5.0, 50.0, float(fat_pct))
                p_freq  = st.number_input("Freq (days/week)", 1, 7, int(frequency))
                p_exp   = st.selectbox("Experience", [1,2,3],
                                        format_func=lambda x:{1:"Beginner",2:"Intermediate",3:"Advanced"}[x],
                                        index=experience-1)
            p_gender = st.radio("Gender", ["Male","Female"], horizontal=True)
            submitted = st.form_submit_button("⚡ Predict Calories")

        if submitted:
            g_enc = 1 if p_gender == "Male" else 0
            reg_in = pd.DataFrame([{
                "Age": p_age, "Weight (kg)": p_weight, "BMI": p_bmi,
                "Fat_Percentage": p_fat, "Avg_BPM": p_bpm,
                "Max_BPM": p_bpm + 20, "Resting_BPM": p_restbpm,
                "Session_Duration (hours)": p_dur,
                "Workout_Frequency (days/week)": p_freq,
                "Experience_Level": p_exp, "Gender_Encoded": g_enc, "Workout_Type_Encoded": 0,
            }])
            cal = state["rf"].predict(reg_in)[0]
            st.markdown(f"""
            <div class="result-card" style="text-align:center">
                <div style="color:#adb5bd;font-size:0.9rem">Estimated Calorie Burn</div>
                <div class="metric-value" style="font-size:3rem">{int(cal)}</div>
                <div style="color:#adb5bd">kcal per session</div>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — ADAPTIVE FEEDBACK
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown('<div class="section-header">🔄 Adaptive Feedback System</div>', unsafe_allow_html=True)

        if st.session_state["result"] is None:
            st.markdown('<div class="warn-box">⚠️ Please generate your plan first (Tab: My Plan).</div>',
                        unsafe_allow_html=True)
        else:
            r = st.session_state["result"]
            st.markdown(f"""
            <div class="info-box">
                Adjusting plan for: <strong>{r['Fitness_Goal']}</strong> &nbsp;|&nbsp;
                Current intensity: <strong>{r['Intensity']}</strong>
            </div>
            """, unsafe_allow_html=True)

            with st.form("feedback_form"):
                fc1, fc2, fc3 = st.columns(3)
                with fc1:
                    difficulty = st.select_slider(
                        "How difficult was the plan?",
                        options=[1, 2, 3],
                        format_func=lambda x: {1:"Too Easy 😌", 2:"Just Right 😊", 3:"Too Hard 😅"}[x],
                        value=2,
                    )
                with fc2:
                    sessions = st.slider("Sessions completed this week", 0, 7, 5)
                with fc3:
                    w_change = st.number_input("Weight change (kg, − = loss)", -10.0, 10.0, -0.5, 0.1)

                fb_submitted = st.form_submit_button("📊 Get Feedback Report")

            if fb_submitted:
                adj = adaptive_feedback(r, {
                    "perceived_difficulty": difficulty,
                    "sessions_completed": sessions,
                    "weight_change_kg": w_change,
                })
                score = adj["Performance_Score"]

                c1, c2 = st.columns([1, 2])
                with c1:
                    pct = int(score)
                    st.markdown(f"""
                    <div style="text-align:center;margin:1.5rem 0">
                        <div style="font-size:0.85rem;color:#adb5bd;margin-bottom:0.5rem;text-transform:uppercase;letter-spacing:0.05em">
                            Weekly Performance Score
                        </div>
                        <div style="font-size:4rem;font-weight:800;color:{'#51cf66' if score>=70 else '#ffa94d' if score>=40 else '#e94560'}">
                            {score}<span style="font-size:1.5rem">/100</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(score / 100)
                    st.markdown(f"""
                    <div class="info-box" style="margin-top:1rem">
                        <strong>Next Week:</strong><br>{adj['Next_Week_Goal']}
                    </div>
                    """, unsafe_allow_html=True)

                with c2:
                    st.markdown("#### 📋 Recommended Adjustments")
                    for a in adj["Adjustments"]:
                        st.markdown(f'<div class="adj-item">{a}</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 — PROGRESS DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown('<div class="section-header">📈 12-Week Progress Simulator</div>', unsafe_allow_html=True)

        if st.session_state["result"] is None:
            start_weight = float(weight)
            start_fat    = float(fat_pct)
            start_bmi    = round(weight / height**2, 2)
        else:
            r = st.session_state["result"]
            start_weight = float(weight)
            start_fat    = float(fat_pct)
            start_bmi    = r["BMI"]

        np.random.seed(42)
        weeks = list(range(1, 13))
        prog = {
            "Week":           weeks,
            "Weight_kg":      [start_weight - i*0.35 + np.random.normal(0,0.2) for i in range(12)],
            "BMI":            [start_bmi - i*0.11 + np.random.normal(0,0.08) for i in range(12)],
            "Fat_Pct":        [start_fat - i*0.28 + np.random.normal(0,0.15) for i in range(12)],
            "Calories_Burned":[650 + i*18 + np.random.normal(0,25) for i in range(12)],
            "Performance":    [58 + i*3.2 + np.random.normal(0,2) for i in range(12)],
        }
        prog_df = pd.DataFrame(prog)

        # Summary metrics
        w_lost  = start_weight - prog_df["Weight_kg"].iloc[-1]
        bmi_red = start_bmi - prog_df["BMI"].iloc[-1]
        fat_red = start_fat - prog_df["Fat_Pct"].iloc[-1]
        avg_perf = prog_df["Performance"].mean()

        c1, c2, c3, c4 = st.columns(4)
        for col, val, label in [
            (c1, f"{w_lost:.1f} kg", "Weight Lost"),
            (c2, f"{bmi_red:.2f}", "BMI Reduction"),
            (c3, f"{fat_red:.1f}%", "Fat % Lost"),
            (c4, f"{avg_perf:.0f}/100", "Avg Performance"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:#51cf66">{val}</div>
                    <div class="metric-label">{label}</div>
                    <div class="metric-sub">over 12 weeks</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        fig_dash = make_subplots(rows=2, cols=3,
            subplot_titles=["Weight (kg)", "BMI Trend", "Body Fat %",
                            "Calories Burned / Session", "Performance Score", "Goal Progress %"])

        trace_cfg = [
            ("Weight_kg","#e94560",1,1),("BMI","#ffa94d",1,2),
            ("Fat_Pct","#51cf66",1,3),("Calories_Burned","#339af0",2,1),
            ("Performance","#cc5de8",2,2),
        ]
        for col, clr, r_, c_ in trace_cfg:
            fig_dash.add_trace(go.Scatter(x=prog_df["Week"], y=prog_df[col],
                                          mode="lines+markers", marker_color=clr,
                                          showlegend=False), row=r_, col=c_)

        goal_pct = min(100, (w_lost / 5) * 100)
        fig_dash.add_trace(go.Bar(x=["Fat Loss Goal"], y=[goal_pct],
                                   marker_color="#51cf66", showlegend=False), row=2, col=3)

        fig_dash.update_layout(height=580, template="plotly_dark",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               title="🏋️ FitGenie AI — 12-Week Progress Dashboard",
                               title_font_color="#fff")
        st.plotly_chart(fig_dash, use_container_width=True)

        with st.expander("📊 Week-by-Week Data Table"):
            disp = prog_df.copy()
            disp.columns = ["Week","Weight (kg)","BMI","Fat %","Cal Burned","Performance"]
            st.dataframe(disp.round(2), use_container_width=True)

    # Footer
    st.markdown("""
    <hr style="border-color:#2d2d45;margin-top:3rem">
    <div style="text-align:center;color:#6c757d;font-size:0.8rem;padding:1rem">
        🏋️ FitGenie AI &nbsp;|&nbsp; K-Means Clustering · Random Forest · XGBoost · Classification
        &nbsp;|&nbsp; Built with Streamlit &amp; Scikit-Learn
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
