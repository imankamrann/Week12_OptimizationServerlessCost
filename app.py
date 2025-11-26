
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from io import StringIO

# ========================================
# PAGE CONFIG
# ========================================
st.set_page_config(
    page_title="Serverless FinOps Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# TITLE
# ========================================
st.title("FinOps at Scale for Serverless Applications")
st.markdown("**INFO 49971 – Cloud Economics | Serverless Cost Analysis**")

# ========================================
# DATA LOADING
# ========================================
data_folder = "data"

if not os.path.exists(data_folder):
    st.error(f"Folder '{data_folder}' not found. Please create it and add the CSV file.")
    st.stop()

# --- Load Dataset ---
dataset_path = os.path.join(data_folder, "Serverless_Data.csv")
if not os.path.exists(dataset_path):
    st.error(f"Dataset file not found: {dataset_path}")
    st.stop()

# Read and clean the CSV to handle potential quoted lines
with open(dataset_path, 'r') as f:
    lines = f.readlines()

clean_lines = [line.strip().strip('"') for line in lines]
csv_string = '\n'.join(clean_lines)
df = pd.read_csv(StringIO(csv_string))

st.success(f"Loaded dataset ({len(df):,} rows).")

# ========================================
# DATA CLEANING
# ========================================
# No missing values based on info, but fill if any
df = df.fillna(0)

# Add calculated columns if needed
df['AvgDurationSec'] = df['AvgDurationMs'] / 1000
df['InvocationPercent'] = df['InvocationsPerMonth'] / df['InvocationsPerMonth'].sum() * 100

# ========================================
# SIDEBAR FILTERS
# ========================================
st.sidebar.header("Filters")

environments = st.sidebar.multiselect("Environment", df["Environment"].unique(), default=df["Environment"].unique())
memory_levels = st.sidebar.multiselect("Memory (MB)", sorted(df["MemoryMB"].unique()), default=sorted(df["MemoryMB"].unique()))
invocation_min, invocation_max = st.sidebar.slider("Invocations Per Month", int(df['InvocationsPerMonth'].min()), int(df['InvocationsPerMonth'].max()), (int(df['InvocationsPerMonth'].min()), int(df['InvocationsPerMonth'].max())))
duration_min, duration_max = st.sidebar.slider("Avg Duration (ms)", int(df['AvgDurationMs'].min()), int(df['AvgDurationMs'].max()), (int(df['AvgDurationMs'].min()), int(df['AvgDurationMs'].max())))

# Apply Filters
mask = (
    df["Environment"].isin(environments) &
    df["MemoryMB"].isin(memory_levels) &
    (df["InvocationsPerMonth"] >= invocation_min) &
    (df["InvocationsPerMonth"] <= invocation_max) &
    (df["AvgDurationMs"] >= duration_min) &
    (df["AvgDurationMs"] <= duration_max)
)
filtered_df = df[mask].copy()

# ========================================
# COMPUTATIONS
# ========================================
total_functions = len(filtered_df)
total_cost = filtered_df['CostUSD'].sum()
avg_cold_start = filtered_df['ColdStartRate'].mean()
total_gb_seconds = filtered_df['GBSeconds'].sum()

# Exercise 1: Top cost contributors
sorted_df = filtered_df.sort_values('CostUSD', ascending=False)
sorted_df['CumulativeCost'] = sorted_df['CostUSD'].cumsum()
sorted_df['CumulativePercent'] = sorted_df['CumulativeCost'] / total_cost * 100
pareto_functions = sorted_df[sorted_df['CumulativePercent'] <= 80]

# Exercise 2: Memory right-sizing candidates (low duration, high memory)
memory_candidates = filtered_df[(filtered_df['AvgDurationMs'] < 1000) & (filtered_df['MemoryMB'] > 1024)]

# Exercise 3: Provisioned concurrency
pc_df = filtered_df[filtered_df['ProvisionedConcurrency'] > 0]

# Exercise 4: Unused/low-value
unused_df = filtered_df[(filtered_df['InvocationPercent'] < 1) & (filtered_df['CostUSD'] > filtered_df['CostUSD'].median())]

# Exercise 6: Container candidates
container_candidates = filtered_df[(filtered_df['AvgDurationMs'] > 3000) & (filtered_df['MemoryMB'] > 2048) & (filtered_df['InvocationsPerMonth'] < 100000)]

# ========================================
# KPI CARDS
# ========================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Functions", total_functions)
col2.metric("Total Cost (USD)", f"${total_cost:.2f}")
col3.metric("Avg Cold Start Rate", f"{avg_cold_start:.2%}")
col4.metric("Total GB-Seconds", f"{total_gb_seconds:.2f}")

# ========================================
# TABS
# ========================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview", "Cost Contributors", "Memory Right-Sizing", "Concurrency Optimization",
    "Unused Workloads", "Cost Forecasting", "Container Candidates"
])

# ---------- TAB 1: Overview ----------
with tab1:
    st.subheader("Dataset Overview")
    st.write("**First 5 Rows**")
    st.dataframe(filtered_df.head())

    st.write("**Summary Statistics**")
    st.write(filtered_df.describe())

    st.write("**Missing Values**")
    st.write(filtered_df.isnull().sum())

# ---------- TAB 2: Cost Contributors (Exercise 1) ----------
with tab2:
    st.subheader("Top Cost Contributors")
    st.write("Functions contributing ~80% of total spend:")
    st.dataframe(pareto_functions[['FunctionName', 'Environment', 'CostUSD', 'CumulativePercent']])

    st.write("**Cost vs Invocation Frequency**")
    fig_scatter = px.scatter(filtered_df, x='InvocationsPerMonth', y='CostUSD', color='Environment', hover_name='FunctionName', title="Cost vs Invocations")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ---------- TAB 3: Memory Right-Sizing (Exercise 2) ----------
with tab3:
    st.subheader("Memory Right-Sizing Opportunities")
    st.write("Functions with low duration (<1000ms) but high memory (>1024MB):")
    st.dataframe(memory_candidates[['FunctionName', 'AvgDurationMs', 'MemoryMB', 'CostUSD']])

    st.write("**Predict Cost Impact**")
    selected_func = st.selectbox("Select Function", memory_candidates['FunctionName'])
    func_row = filtered_df[filtered_df['FunctionName'] == selected_func].iloc[0]
    new_memory = st.slider("New Memory (MB)", 128, int(func_row['MemoryMB']), int(func_row['MemoryMB']) // 2, step=128)
    # Assume cost proportional to memory (simple model, ignoring duration change)
    original_cost = func_row['CostUSD']
    predicted_cost = original_cost * (new_memory / func_row['MemoryMB'])
    st.metric("Predicted New Cost", f"${predicted_cost:.2f}", delta=f"{(predicted_cost - original_cost):.2f}")

# ---------- TAB 4: Concurrency Optimization (Exercise 3) ----------
with tab4:
    st.subheader("Provisioned Concurrency Optimization")
    st.write("Functions with Provisioned Concurrency:")
    st.dataframe(pc_df[['FunctionName', 'ColdStartRate', 'ProvisionedConcurrency', 'CostUSD']])

    st.write("**Cold Start Rate vs PC Cost**")
    fig_bar = px.bar(pc_df, x='FunctionName', y='ProvisionedConcurrency', color='ColdStartRate', title="PC vs Cold Start")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.write("**Recommendations**")
    for _, row in pc_df.iterrows():
        if row['ColdStartRate'] < 0.01:
            st.write(f"- {row['FunctionName']}: Low cold starts ({row['ColdStartRate']:.2%}), consider removing PC to save costs.")

# ---------- TAB 5: Unused Workloads (Exercise 4) ----------
with tab5:
    st.subheader("Unused or Low-Value Workloads")
    st.write("Functions with <1% of total invocations but above median cost:")
    st.dataframe(unused_df[['FunctionName', 'InvocationPercent', 'CostUSD']])

# ---------- TAB 6: Cost Forecasting (Exercise 5) ----------
with tab6:
    st.subheader("Cost Forecasting Model")
    st.write("Simple Model: Cost ≈ (Invocations / 1e6 * 0.20) + (GBSeconds * 0.000016667) + (DataTransferGB * 0.09)")
    # Using approximate AWS pricing for illustration

    selected_func_forecast = st.selectbox("Select Function for Forecast", filtered_df['FunctionName'])
    func_row_forecast = filtered_df[filtered_df['FunctionName'] == selected_func_forecast].iloc[0]

    new_invocations = st.number_input("Projected Invocations", value=int(func_row_forecast['InvocationsPerMonth']))
    new_duration = st.number_input("Projected Avg Duration (ms)", value=int(func_row_forecast['AvgDurationMs']))
    new_memory = st.number_input("Projected Memory (MB)", value=int(func_row_forecast['MemoryMB']))
    new_data_transfer = st.number_input("Projected Data Transfer (GB)", value=int(func_row_forecast['DataTransferGB']))

    new_gb_seconds = (new_invocations * (new_duration / 1000) * (new_memory / 1024))
    predicted_cost_forecast = (new_invocations / 1e6 * 0.20) + (new_gb_seconds * 0.000016667) + (new_data_transfer * 0.09)
    st.metric("Forecasted Cost", f"${predicted_cost_forecast:.2f}")

# ---------- TAB 7: Container Candidates (Exercise 6) ----------
with tab7:
    st.subheader("Workloads for Containerization")
    st.write("Long-running (>3s), High Memory (>2GB), Low Invocations (<100k/month):")
    st.dataframe(container_candidates[['FunctionName', 'AvgDurationSec', 'MemoryMB', 'InvocationsPerMonth', 'CostUSD']])

# ========================================
# DOWNLOAD BUTTONS
# ========================================
st.sidebar.markdown("---")
st.sidebar.download_button("Download Filtered Data", filtered_df.to_csv(index=False).encode(), "filtered_serverless.csv", "text/csv")

# ========================================
# FOOTER
# ========================================
st.markdown("---")
st.caption("Built with Streamlit • Fall 2025 • Sheridan College")
