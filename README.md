# 🏋️ FitGenie AI — Streamlit Web App

Personalized Fitness Recommendation System powered by K-Means Clustering, Random Forest, and Classification ML models.

## 🚀 Deploy to Streamlit Community Cloud (FREE — shareable link)

### Step 1 — Push to GitHub
1. Create a **new public GitHub repository** (e.g. `fitgenie-ai`)
2. Upload both files into the repo root:
   - `app.py`
   - `requirements.txt`

### Step 2 — Deploy on Streamlit Cloud
1. Go to **https://share.streamlit.io**
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repo → branch `main` → main file `app.py`
5. Click **Deploy** — your app will be live in ~2 minutes at:
   ```
   https://<your-username>-fitgenie-ai-app-<hash>.streamlit.app
   ```
6. **Share the link** with anyone — no login required to view!

---

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501

---

## ✨ Features Implemented

| Module | Description |
|--------|-------------|
| **User Profile** | Sidebar inputs: age, gender, weight, height, BPM, experience, fat %, frequency |
| **My Plan (Tab 1)** | BMI, goal banner, 7-day workout plan, nutrition tips, stat cards |
| **Data Insights (Tab 2)** | Distribution plots, pie chart, box plot, correlation heatmap |
| **Clustering (Tab 3)** | Elbow + silhouette analysis, PCA 2D scatter, cluster profile table |
| **Calorie Predictor (Tab 4)** | Model comparison (LR vs RF), actual vs predicted, feature importance, interactive predictor |
| **Adaptive Feedback (Tab 5)** | Performance score, difficulty adjustment, session adherence, goal-specific tips |
| **Progress Dashboard (Tab 6)** | 12-week simulated progress: weight, BMI, fat%, calories, performance |
