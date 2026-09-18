
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="EcoPower AI",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ EcoPower AI")
st.caption("Upload your electricity consumption data and let AI spot unusual usage for you.")

# ----------------------------------------------------------------------
# Sidebar — instructions & settings
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("How it works")
    st.markdown(
        """
        1. **Upload** a CSV file with your electricity usage.
        2. EcoPower AI uses a machine-learning model called
           **Isolation Forest** to find readings that look unusual
           compared to the rest of your data.
        3. You get **charts** and **easy tips** to help you save energy.
        """
    )
    st.divider()
    st.subheader("CSV format expected")
    st.code(
        "timestamp, consumption_kwh\n"
        "2024-01-01 00:00, 1.2\n"
        "2024-01-01 01:00, 1.1\n"
        "2024-01-01 02:00, 5.8   <- unusual spike\n"
        "...",
        language="text",
    )
    st.divider()
    st.subheader("Model sensitivity")
    contamination = st.slider(
        "How strict should anomaly detection be?",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
        help=(
            "This tells the model roughly what fraction of readings to "
            "flag as unusual. Lower = stricter (fewer flags), "
            "Higher = looser (more flags)."
        ),
    )

# ----------------------------------------------------------------------
# File upload
# ----------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload your electricity consumption CSV",
    type=["csv"],
    help="The file should have a timestamp column and a consumption column (in kWh).",
)

if uploaded_file is None:
    st.info(
        "👋 Upload a CSV to get started. Not sure what that looks like? "
        "Check the sidebar for an example format."
    )
    st.stop()

# ----------------------------------------------------------------------
# Load & clean the data
# ----------------------------------------------------------------------
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Couldn't read that file as a CSV. Error: {e}")
    st.stop()

if df.empty:
    st.error("The uploaded file is empty. Please upload a CSV with some data in it.")
    st.stop()

st.subheader("1️⃣ Preview of your data")
st.dataframe(df.head(10), use_container_width=True)

# --- Let the user tell us which columns are which, in case names differ ---
cols = list(df.columns)

# Try to auto-guess sensible defaults
def guess_column(possible_names, columns):
    for name in possible_names:
        for c in columns:
            if name in c.lower():
                return c
    return columns[0]

guessed_time_col = guess_column(["time", "date", "timestamp"], cols)
guessed_usage_col = guess_column(["kwh", "consum", "usage", "energy", "power"], cols)

col1, col2 = st.columns(2)
with col1:
    time_col = st.selectbox(
        "Which column is the date/time?",
        options=cols,
        index=cols.index(guessed_time_col),
    )
with col2:
    usage_col = st.selectbox(
        "Which column is the electricity consumption (kWh)?",
        options=cols,
        index=cols.index(guessed_usage_col),
    )

# Clean up the working dataframe
work_df = df[[time_col, usage_col]].copy()
work_df.columns = ["timestamp", "consumption_kwh"]

# Convert types, drop rows we can't use
work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], errors="coerce")
work_df["consumption_kwh"] = pd.to_numeric(work_df["consumption_kwh"], errors="coerce")
before_rows = len(work_df)
work_df = work_df.dropna(subset=["timestamp", "consumption_kwh"]).sort_values("timestamp")
after_rows = len(work_df)

if after_rows < before_rows:
    st.warning(
        f"Removed {before_rows - after_rows} row(s) with missing or unreadable "
        "date/consumption values."
    )

if len(work_df) < 10:
    st.error(
        "Not enough valid data to analyze (need at least 10 readings). "
        "Please check your file and column selections."
    )
    st.stop()

# ----------------------------------------------------------------------
# Run Isolation Forest
# ----------------------------------------------------------------------
st.subheader("2️⃣ Detecting unusual consumption")

with st.spinner("Training the anomaly detection model..."):
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
    )
    X = work_df[["consumption_kwh"]].values
    work_df["anomaly_flag"] = model.fit_predict(X)  # -1 = anomaly, 1 = normal
    work_df["anomaly_score"] = model.decision_function(X)  # lower = more unusual
    work_df["is_anomaly"] = work_df["anomaly_flag"] == -1

n_anomalies = int(work_df["is_anomaly"].sum())
pct_anomalies = 100 * n_anomalies / len(work_df)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total readings", f"{len(work_df):,}")
m2.metric("Unusual readings found", f"{n_anomalies:,}")
m3.metric("% flagged unusual", f"{pct_anomalies:.1f}%")
m4.metric("Average consumption", f"{work_df['consumption_kwh'].mean():.2f} kWh")

st.caption(
    "🔍 'Unusual' readings are ones that look very different from your typical "
    "usage pattern — much higher or oddly timed compared to the rest of your data."
)

# ----------------------------------------------------------------------
# Graphs
# ----------------------------------------------------------------------
st.subheader("3️⃣ Visualizing your consumption")

tab1, tab2, tab3 = st.tabs(["📈 Timeline", "📊 Distribution", "🗓️ Daily pattern"])

with tab1:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=work_df["timestamp"],
            y=work_df["consumption_kwh"],
            mode="lines",
            name="Consumption (kWh)",
            line=dict(color="#2E86AB", width=1.5),
        )
    )
    anomalies = work_df[work_df["is_anomaly"]]
    fig.add_trace(
        go.Scatter(
            x=anomalies["timestamp"],
            y=anomalies["consumption_kwh"],
            mode="markers",
            name="Unusual reading",
            marker=dict(color="#E63946", size=9, symbol="circle-open", line=dict(width=2)),
        )
    )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Consumption (kWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=450,
        margin=dict(t=40),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig2 = px.histogram(
        work_df,
        x="consumption_kwh",
        color=work_df["is_anomaly"].map({True: "Unusual", False: "Normal"}),
        nbins=40,
        color_discrete_map={"Unusual": "#E63946", "Normal": "#2E86AB"},
        labels={"consumption_kwh": "Consumption (kWh)", "color": "Reading type"},
    )
    fig2.update_layout(height=450, legend_title_text="Reading type", margin=dict(t=40))
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    daily_df = work_df.copy()
    daily_df["hour"] = daily_df["timestamp"].dt.hour
    hourly_avg = daily_df.groupby("hour")["consumption_kwh"].mean().reset_index()
    fig3 = px.bar(
        hourly_avg,
        x="hour",
        y="consumption_kwh",
        labels={"hour": "Hour of day", "consumption_kwh": "Average consumption (kWh)"},
        color_discrete_sequence=["#2E86AB"],
    )
    fig3.update_layout(height=450, margin=dict(t=40))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("This shows your average usage by hour of day, across all the dates in your file.")

# ----------------------------------------------------------------------
# Flagged readings table
# ----------------------------------------------------------------------
with st.expander("🔎 See the exact readings flagged as unusual"):
    if n_anomalies == 0:
        st.write("No unusual readings were found with the current sensitivity setting.")
    else:
        show_df = anomalies[["timestamp", "consumption_kwh", "anomaly_score"]].sort_values(
            "anomaly_score"
        )
        show_df.columns = ["Timestamp", "Consumption (kWh)", "Unusualness score (lower = more unusual)"]
        st.dataframe(show_df, use_container_width=True)

# ----------------------------------------------------------------------
# Simple recommendations
# ----------------------------------------------------------------------
st.subheader("4️⃣ Energy-saving recommendations")

avg_usage = work_df["consumption_kwh"].mean()
peak_hour = int(
    work_df.assign(hour=work_df["timestamp"].dt.hour)
    .groupby("hour")["consumption_kwh"]
    .mean()
    .idxmax()
)
night_hours = work_df[work_df["timestamp"].dt.hour.isin([0, 1, 2, 3, 4])]
night_avg = night_hours["consumption_kwh"].mean() if not night_hours.empty else 0

tips = []

if n_anomalies > 0:
    tips.append(
        f"⚠️ We found **{n_anomalies} unusual reading(s)** ({pct_anomalies:.1f}% of your data). "
        "These spikes could be caused by a specific appliance being left on, a faulty device, "
        "or an unusually high-usage day. Check what was running during those times."
    )
else:
    tips.append(
        "✅ No major unusual spikes were detected — your consumption pattern looks fairly consistent."
    )

tips.append(
    f"🕐 Your consumption tends to peak around **{peak_hour}:00**. Consider shifting flexible, "
    "high-energy tasks (laundry, dishwasher, EV charging) to off-peak hours if your utility "
    "offers time-of-use pricing."
)

if night_avg > 0 and night_avg > 0.5 * avg_usage:
    tips.append(
        "🌙 Your overnight usage (12am–4am) is relatively high compared to your average. "
        "This can indicate devices left on standby, always-on appliances, or HVAC systems "
        "running more than needed — worth double-checking."
    )

tips.append(
    "💡 General habits that help: switch to LED bulbs, unplug idle electronics, "
    "set your thermostat 1–2 degrees closer to outdoor temperature, and run "
    "full loads in washers/dishwashers instead of partial ones."
)

tips.append(
    "📊 Revisit this dashboard weekly or monthly — comparing new uploads over time will help "
    "you see whether changes you make are actually reducing unusual and overall usage."
)

for tip in tips:
    st.markdown(f"- {tip}")

st.divider()
st.caption(
    "EcoPower AI uses an unsupervised machine-learning model (Isolation Forest) to statistically "
    "flag unusual readings. It does not know your specific appliances or habits — use these "
    "insights as a starting point for further investigation, not a guaranteed diagnosis."
)
