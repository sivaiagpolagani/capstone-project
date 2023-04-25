import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


# Load the dataset into a Pandas DataFrame
df = pd.read_csv('ardd_fatalities.csv')

st.table(df.head())
st.table(df.tail())


# Perform data exploration
st.subheader("Data Exploration")
st.write("Dataset shape: ", df.shape)
st.write("Columns: ", df.columns)
st.write("Data types: ", df.dtypes)
st.write("Summary statistics: ", df.describe())
st.write("Missing values: ", df.isnull().sum())




# Create visualizations using Matplotlib and Seaborn
st.subheader("Data Visualization")

# Bar chart to visualize the distribution of road fatalities by gender
plt.figure(figsize=(8, 6))
sns.countplot(x='Gender', data=df)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Distribution of Road Fatalities by Gender')
st.pyplot(plt)



