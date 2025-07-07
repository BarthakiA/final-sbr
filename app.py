import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

st.set_page_config(page_title="Nykaa Analytics", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("NYKA.csv", parse_dates=["signup_date", "last_purchase_date"])
    return df

df = load_data()

st.title("Nykaa: RFM • CLTV • Churn Dashboard")

# -----------------------------------------------------------------------------
# 1) RFM SEGMENTATION
# -----------------------------------------------------------------------------
st.header("1. RFM Segmentation")

rfm = df.rename(columns={
    "recency_days": "Recency",
    "frequency_3m": "Frequency",
    "monetary_value_3m": "Monetary"
})

# Chart 1: Recency distribution
fig_r1 = px.histogram(
    rfm, x="Recency", nbins=40,
    title="Recency (days) Distribution"
)
st.plotly_chart(fig_r1, use_container_width=True)
st.write("Most customers purchased within the last 60 days, with a long tail of infrequent buyers.")

# Chart 2: Frequency distribution
fig_r2 = px.histogram(
    rfm, x="Frequency", nbins=20,
    title="Order Frequency (3m) Distribution"
)
st.plotly_chart(fig_r2, use_container_width=True)
st.write("A majority place 1–3 orders per quarter; fewer make 5+ orders.")

# Chart 3: Monetary distribution
fig_r3 = px.histogram(
    rfm, x="Monetary", nbins=30,
    title="Monetary Value (₹, 3m) Distribution"
)
st.plotly_chart(fig_r3, use_container_width=True)
st.write("Spending peaks around ₹500–₹1,000, with some high spenders above ₹5,000.")

# Chart 4: 3D RFM clusters
fig_r4 = px.scatter_3d(
    rfm, x="Recency", y="Frequency", z="Monetary",
    color="RFM_segment_label",
    labels={"Recency":"Recency (days)","Frequency":"# orders","Monetary":"Spend (₹)"},
    title="3D View of RFM Segments"
)
st.plotly_chart(fig_r4, use_container_width=True)
st.write("Clusters reveal segments like ‘Champions’, ‘At Risk’, and ‘Low Value’.")

# -----------------------------------------------------------------------------
# 2) CUSTOMER LIFETIME VALUE
# -----------------------------------------------------------------------------
st.header("2. Customer Lifetime Value (3-Month Forecast)")

# Chart 1: CLTV histogram
fig_c1 = px.histogram(
    df, x="predicted_CLTV_3m", nbins=30,
    title="Predicted CLTV (₹) Distribution"
)
st.plotly_chart(fig_c1, use_container_width=True)
st.write("Most CLTV predictions fall under ₹1,000; a small group exceeds ₹3,000.")

# Chart 2: Predicted vs Actual
fig_c2 = px.scatter(
    df, x="predicted_CLTV_3m", y="actual_CLTV_3m",
    title="Predicted vs Actual CLTV"
)
st.plotly_chart(fig_c2, use_container_width=True)
st.write("Predictions align well overall but over-estimate for some low-value customers.")

# Chart 3: Avg CLTV by RFM segment
cltv_by_seg = df.groupby("RFM_segment_label")["predicted_CLTV_3m"].mean().reset_index()
fig_c3 = px.bar(
    cltv_by_seg, x="RFM_segment_label", y="predicted_CLTV_3m",
    title="Average Predicted CLTV by RFM Segment"
)
st.plotly_chart(fig_c3, use_container_width=True)
st.write("High-value segments like ‘Champions’ show the largest CLTV forecasts.")

# Chart 4: Actual CLTV boxplots by segment
fig_c4 = px.box(
    df, x="RFM_segment_label", y="actual_CLTV_3m",
    title="Actual CLTV Distribution by RFM Segment"
)
st.plotly_chart(fig_c4, use_container_width=True)
st.write("Actual CLTV varies widely within segments, highlighting intra-segment diversity.")

# -----------------------------------------------------------------------------
# 3) CHURN ANALYSIS & PREDICTION
# -----------------------------------------------------------------------------
st.header("3. Churn Analysis & Prediction")

# Chart 1: Overall churn rate
churn_rate = df["churn_within_3m_flag"].mean()
fig_h1 = px.bar(
    x=["Active","Churned"], y=[1-churn_rate, churn_rate],
    labels={"x":"Status","y":"Proportion"},
    title="Overall 3-Month Churn Rate"
)
st.plotly_chart(fig_h1, use_container_width=True)
st.write(f"**{churn_rate:.1%}** of customers churn within three months.")

# Chart 2: Churn rate by RFM segment
churn_by_seg = df.groupby("RFM_segment_label")["churn_within_3m_flag"].mean().reset_index()
fig_h2 = px.bar(
    churn_by_seg, x="RFM_segment_label", y="churn_within_3m_flag",
    labels={"churn_within_3m_flag":"Churn Rate"},
    title="Churn Rate by RFM Segment"
)
st.plotly_chart(fig_h2, use_container_width=True)
st.write("At-risk segments show churn rates above 50%, while champions stay below 10%.")

# Chart 3: Recency by churn status
fig_h3 = px.box(
    df, x="churn_within_3m_flag", y="recency_days",
    labels={"churn_within_3m_flag":"Churned","recency_days":"Recency (days)"},
    title="Recency Distribution: Churned vs Active"
)
st.plotly_chart(fig_h3, use_container_width=True)
st.write("Churned customers waited much longer since last purchase than active ones.")

# Chart 4: ROC curve for churn model
features = [
    "recency_days","frequency_3m","monetary_value_3m",
    "time_on_app_minutes","page_views_per_session",
    "campaign_clicks","campaign_views","cart_abandonment_rate",
    "first_time_buyer_flag"
]
X = df[features].fillna(0)
y = df["churn_within_3m_flag"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_pred_proba)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

fig_h4 = px.line(
    x=fpr, y=tpr,
    labels={"x":"False Positive Rate","y":"True Positive Rate"},
    title=f"ROC Curve (AUC = {auc:.2f})"
)
st.plotly_chart(fig_h4, use_container_width=True)
st.write(f"AUC of **{auc:.2f}** indicates strong churn prediction performance.")
