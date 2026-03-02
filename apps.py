import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

# PAGE CONFIG
st.set_page_config(
    page_title="Cinematic AI Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CUSTOM CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"] {
    font-family: 'Orbitron', sans-serif;
    color: white;
    background: #0f0c29;
    overflow-x: hidden;
}

body {
    background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #000000);
    background-size: 400% 400%;
    animation: gradientBG 18s ease infinite;
}

@keyframes gradientBG {
    0% {background-position:0% 50%}
    50% {background-position:100% 50%}
    100% {background-position:0% 50%}
}

section[data-testid="stSidebar"] {
    display: none;
}

.neon-title {
    font-size: 48px;
    text-align: center;
    font-weight: 800;
    margin-bottom: 15px;
    text-shadow: 0 0 20px #00f0ff;
}

/* ===== KPI BAR ===== */
.kpi-container {
    display: flex;
    gap: 25px;
    margin: 30px 0;
}

.kpi-card {
    flex: 1;
    padding: 16px;
    border-radius: 18px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(15px);
    border: 2px solid transparent;
    background-clip: padding-box;
    position: relative;
    display: flex;                 
    flex-direction: column;        
    align-items: center;          
    justify-content: center;      
    text-align: center;
    text-align: center; 
}

.kpi-card::before {
    content: "";
    position: absolute;
    inset: -2px;
    border-radius: 20px;
    padding: 2px;
    background: linear-gradient(45deg,#00f0ff,#ff00ff);
    -webkit-mask:
        linear-gradient(#000 0 0) content-box,
        linear-gradient(#000 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
}

.kpi-title {
    font-size: 18px;
    opacity: 0.9;
}

.kpi-value {
    font-size: 27px;
    margin-top: 10px;
}

.kpi-card h2 {
     margin: 0;
    font-size: 32px;
    font-weight: 800;

    background: linear-gradient(90deg, #00f0ff, #ff00ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    text-shadow: 0 0 15px rgba(255, 0, 255, 0.6);
}

.kpi-card p {
    margin-top: 10px;
    font-size: 20px; 
    opacity: 0.9;
    text-align: center;  
}
                        
.movie-slider {
    display: flex;
    overflow-x: auto;
    gap: 30px;
    padding: 40px 20px;
    border-radius: 25px;
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(15px);
}

.movie-card {
    min-width: 260px;
    background: linear-gradient(145deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
    padding: 25px;
    border-radius: 20px;
    transition: 0.4s ease;
    box-shadow: 0 0 25px rgba(0,255,255,0.3);
    flex-shrink: 0;
}

.movie-card:hover {
    transform: scale(1.08);
    box-shadow: 0 0 60px #ff00ff;
}

.movie-card h4 {
    font-size: 17px;
    margin-bottom: 20px;
}

.rating {
    font-size: 15px;
    color: gold;
    text-shadow: 0 0 12px gold;
}

.stButton>button {
    background: linear-gradient(90deg, #00f0ff, #ff00ff);
    color: white;
    border-radius: 30px;
    padding: 10px 25px;
    border: none;
    font-weight: bold;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px #ff00ff;
}
</style>
""", unsafe_allow_html=True)

# LOAD MODEL
@st.cache_resource
def load_model():
    return joblib.load("recommender_system1.joblib")

data = load_model()
best_model_name = data["best_model_name"]
best_predictions = data["best_predictions"]
train_matrix = data["train_matrix"]
movie_mapping = data["movie_mapping"]

# HEADER
st.markdown('<div class="neon-title">🎬 Cinematic AI Movie Recommender</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# KPI ROW
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <h2> {train_matrix.shape[0]}</h2>
        <p>👥 Total Users</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <h2> {train_matrix.shape[1]}</h2>
        <p>🎬 Total Movies</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <h2>{best_model_name}</h2>
        <p>🤖 Model</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# USER INPUT
user_list = train_matrix.index.tolist()
selected_user = st.selectbox("Select User", user_list)
n_recs = st.slider("Number of Recommendations", 5, 20, 10)

# RECOMMENDATION FUNCTION
def generate_recommendations(user_id, n=10):
    rated_mask = train_matrix.loc[user_id] > 0
    user_preds = best_predictions.loc[user_id].copy()
    user_preds[rated_mask] = -np.inf
    top_n = user_preds.sort_values(ascending=False).head(n)

    result = []
    for item_id, score in top_n.items():
        title = movie_mapping.get(item_id, f"Movie {item_id}")
        result.append((title, round(score,2)))

    return result

# GENERATE BUTTON
btn_left, btn_center, btn_right = st.columns([6, 5, 6])

with btn_center:
    generate = st.button("🚀 Generate Cinematic Recommendations")

# OUTPUT SECTION
if generate:

    recs = generate_recommendations(selected_user, n_recs)

    if len(recs) == 0:
        st.warning("No recommendations available.")
    else:
        st.markdown(f"### 🎥 Top {n_recs} Picks For You")

    # Slider Container
    movie_html = "<div class='movie-slider'>"

    for title, score in recs:  
        movie_html += f"""
<div class='movie-card'>
    <h4>{title}</h4>
    <div class='rating'>⭐ {round(score,2)}</div>
</div>
"""

    movie_html += "</div>"

    st.markdown(movie_html, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # TOP 10 
    st.markdown("### 🎬 Top 10 Full Recommendations")

    top_10 = recs[:10]

    for title, score in top_10:
        st.markdown(
            f"""
            <div style="
                background: rgba(255,255,255,0.05);
                padding:15px;
                border-radius:15px;
                margin-bottom:10px;
            ">
            🎬 <b>{title}</b>
            <br>
            ⭐ Predicted Rating: {round(score,2)}
            </div>
            """,
            unsafe_allow_html=True
        )
   # Recommendation KPI
    scores = [score for _, score in recs]
    avg_score = sum(scores)/len(scores)
    max_score = max(scores)
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-title">🎬 Total Recommended</div>
            <div class="kpi-value">{len(recs)}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">⭐ Avg Score</div>
            <div class="kpi-value">{avg_score:.2f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">🔥 Top Score</div>
            <div class="kpi-value">{max_score:.2f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Bar Chart
    st.markdown("### 📊 Recommendation Insight")

    rec_df = pd.DataFrame(recs, columns=["Title", "Predicted Rating"])

    # Sort descending 
    rec_df = rec_df.sort_values("Predicted Rating", ascending=True)

    fig = px.bar(
        rec_df,
        x="Predicted Rating",
        y="Title",
        orientation="h",
        text="Predicted Rating",
        color="Predicted Rating",
        color_continuous_scale=["#00f0ff", "#ff00ff"]  # Neon gradient
    )

    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>⭐ Rating: %{x:.2f}<extra></extra>"
    )

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color="grey",
        coloraxis_showscale=False,  
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(title=""),
        xaxis=dict(title="Predicted Rating")
    )

    st.plotly_chart(fig, use_container_width=True)

    # Insight
    st.markdown("### 🧠 AI Insight from Recommendations")
    if avg_score > 4:
        st.success("🔥 Model strongly recommends high-rated movies for this user. User preference pattern is clearly identified.")
    else:
        st.info("📊 Recommendations show moderate confidence. Model still exploring user preference space.")

    confidence_score = avg_score / 5 * 100
    st.write(f"🎯 Personalization Confidence: {confidence_score:.2f}%")

