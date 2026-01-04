import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Page config
st.set_page_config(page_title="IRIS Dataset EDA", layout="wide")

# Title
st.title("🌸 IRIS Dataset – Exploratory Data Analysis")

# Load dataset
@st.cache_data
def load_data():
    return sns.load_dataset("iris")

data = load_data()

# Sidebar
st.sidebar.header("EDA Options")

eda_option = st.sidebar.selectbox(
    "Select Analysis Type",
    [
        "Dataset Overview",
        "Statistical Summary",
        "Distribution Plot",
        "Joint Plot",
        "Pair Plot",
        "Boxen Plot",
        "Strip Plot",
        "Swarm Plot"
    ]
)

# ===================== DATASET OVERVIEW =====================
if eda_option == "Dataset Overview":
    st.subheader("📄 Dataset Preview")
    st.dataframe(data)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", data.shape[0])
    col2.metric("Columns", data.shape[1])
    col3.metric("Missing Values", data.isnull().sum().sum())

    st.subheader("Column Info")
    st.write(data.dtypes)

# ===================== STATISTICS =====================
elif eda_option == "Statistical Summary":
    st.subheader("📊 Descriptive Statistics")
    st.dataframe(data.describe())

# ===================== DISTRIBUTION =====================
elif eda_option == "Distribution Plot":
    st.subheader("📈 Distribution Plot")

    column = st.selectbox(
        "Select Column",
        ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    )

    fig, ax = plt.subplots()
    sns.histplot(data[column], kde=True, ax=ax)
    st.pyplot(fig)

# ===================== JOINT PLOT =====================
elif eda_option == "Joint Plot":
    st.subheader("🔗 Joint Plot")

    x_col = st.selectbox("X Axis", data.columns[:-1])
    y_col = st.selectbox("Y Axis", data.columns[:-1], index=1)
    kind = st.selectbox("Plot Type", ["scatter", "reg", "hex", "kde"])

    fig = sns.jointplot(x=x_col, y=y_col, data=data, kind=kind)
    st.pyplot(fig)

# ===================== PAIR PLOT =====================
elif eda_option == "Pair Plot":
    st.subheader("🔀 Pair Plot (Feature Relationships)")
    fig = sns.pairplot(data)
    st.pyplot(fig)

# ===================== BOXEN PLOT =====================
elif eda_option == "Boxen Plot":
    st.subheader("📦 Boxen Plot")

    x_col = st.selectbox("X Axis", data.columns[:-1])
    y_col = st.selectbox("Y Axis", data.columns[:-1], index=1)

    fig, ax = plt.subplots()
    sns.boxenplot(x=x_col, y=y_col, data=data, ax=ax)
    st.pyplot(fig)

# ===================== STRIP PLOT =====================
elif eda_option == "Strip Plot":
    st.subheader("📌 Strip Plot")

    x_col = st.selectbox("X Axis", data.columns[:-1])
    y_col = st.selectbox("Y Axis", data.columns[:-1], index=1)

    fig, ax = plt.subplots()
    sns.stripplot(x=x_col, y=y_col, data=data, ax=ax)
    st.pyplot(fig)

# ===================== SWARM PLOT =====================
elif eda_option == "Swarm Plot":
    st.subheader("🐝 Swarm Plot")

    x_col = st.selectbox("X Axis", data.columns[:-1])
    y_col = st.selectbox("Y Axis", data.columns[:-1], index=1)

    fig, ax = plt.subplots()
    sns.swarmplot(x=x_col, y=y_col, data=data, ax=ax)
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("✅ **IRIS Dataset EDA using Streamlit & Seaborn**")
