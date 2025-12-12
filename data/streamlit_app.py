
import streamlit as st
import pandas as pd
import sqlite3
import seaborn as sns

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    """
    Loads data from the SQLite database into two pandas DataFrames.
    The function is cached to improve performance.
    """
    # Connect to the database, assuming it's in a 'data' subdirectory
    conn = sqlite3.connect('data/housing_data.db') # MODIFIED PATH HERE
    df_eda = pd.read_sql_query("SELECT * FROM eda_data", conn)
    df_ml = pd.read_sql_query("SELECT * FROM ml_data", conn)
    conn.close()
    return df_eda, df_ml

# Example of how to use the loaded data in a Streamlit app
# if __name__ == "__main__":
#     st.title("Housing Price Analysis App")
#     df_eda, df_ml = load_data()

#     st.subheader("EDA Data Overview")
#     st.write(df_eda.head())

#     st.subheader("ML Data Overview")
#     st.write(df_ml.head())

#     st.success("Data loaded successfully from SQLite database!")
