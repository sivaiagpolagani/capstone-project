import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Load the dataset
df = pd.read_csv('ardd_fatalities.csv')

st.subheader("Data Visualization")

st.subheader("1) Distribution of Road Fatalities by State")
plt.figure(figsize=(8, 6))
sns.countplot(x='State', data=df)
plt.xlabel('State')
plt.ylabel('Count')
plt.title('Distribution of Road Fatalities by State')
st.pyplot(plt)


st.subheader("2) Distribution of Road Fatalities by Road User")
plt.figure(figsize=(20, 12))
sns.countplot(x='Road User', data=df)
plt.xlabel('Road User')
plt.ylabel('Count')
plt.title('Distribution of Road Fatalities by Road User')
st.pyplot(plt)


st.subheader("3) Distribution of Road Fatalities by Age Group")
plt.figure(figsize=(10, 8))
sns.countplot(x='Age Group', data=df)
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.title('Distribution of Road Fatalities by Age Group')
st.pyplot(plt)


st.subheader("4) Comparison of Fatalities on Weekends/Nights vs Weekdays/Daytime")
plt.figure(figsize=(10, 8))
sns.countplot(x='Day of week', hue='Time of day', data=df[(df['Time of day'].isin(['Day'])) | (df['Day of week'].isin(['Weekday'])) & (df['Day of week'].isin(['Weekend'])) | (df['Time of day'].isin(['Night']))])
plt.xlabel('Day of week')
plt.ylabel('Count')
plt.title('Comparison of Fatalities on Weekends/Nights vs Weekdays/Daytime')
st.pyplot(plt)


st.subheader("5) Distribution of Road Fatalities by Road User and Crash Type")
plt.figure(figsize=(15,10))
sns.countplot(x='Road User', hue='Crash Type', data=df)
plt.xlabel('Road User')
plt.ylabel('Count')
plt.title('Distribution of Road Fatalities by Road User and Crash Type')
st.pyplot(plt)


# Data preprocessing
X = df[['Age Group', 'Gender', 'State', 'Time of day', 'Day of week', 'Road User']]
y = df['Crash Type']
X = pd.get_dummies(X) # Convert categorical variables to numerical
X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X, y, test_size=0.2, random_state=42)

# Classification
clasification = SVC()
clasification.fit(X_train_data, y_train_data)
y_prediction = clasification.predict(X_test_data)
accuracy = accuracy_score(y_test_data, y_prediction)

# Display results
st.subheader("Classification (Predictive Analytics)")
st.write("Accuracy (SVM):", accuracy)