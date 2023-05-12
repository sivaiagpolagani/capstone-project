import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# Load the dataset
df = pd.read_csv('ardd_fatalities.csv')

st.subheader("Data Visualization")


st.subheader("1) Distribution of Road Fatalities by State")
plt.figure(figsize=(8, 6))
sns.countplot(x='State', data=df)
plt.xlabel('State')
plt.ylabel('Count')
plt.title('Distribution of Road Fatalities by State')
st.success("The bar graph shows the distribution of road fatalities by state. X-Axis represents the states which are Qld, Vic, SA, WA, NSW, Tas, NT, ACT and Y-Axis represents the number of fatalities count. Moreover, this data is Australian Road Deaths Database March 2023 data. Each state are represented by different colors. NSW is with highest fatalities and ACT with lowest fatalities.")
st.pyplot(plt)




st.subheader("2) Distribution of Road Fatalities by Road User")
plt.figure(figsize=(20, 12))
sns.countplot(x='Road User', data=df)
plt.xlabel('Road User')
plt.ylabel('Count')
plt.title('Distribution of Road Fatalities by Road User')
st.success("The bar graph shows the distribution of road fatalities by Road User. X-Axis represents the type of road users which are Driver, Motorcycle rider, passenger, pedial cyclist and other/-9 and Y-Axis represents the number of fatalities count. Moreover, this data is Australian Road Deaths Database – March 2023 data. Each road users are represented by different colors. Driver is with highest fatalities and other/-9 with lowest fatalities. -9 represents with no data identification.")
st.pyplot(plt)


st.subheader("3) Distribution of Road Fatalities by Age Group")
plt.figure(figsize=(10, 8))
sns.countplot(x='Age Group', data=df)
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.title('Distribution of Road Fatalities by Age Group')
st.success("The bar graph shows the distribution of road fatalities by age group. X-Axis represents the age group which are 40-64, 26-39, 17-25, 65-74, 0-16, 75 or older, -9 and Y-Axis represents the count fatalities. Moreover, this data is Australian Road fatalities by age group – March 2023 data. Each age group are represented by different colors. 40-64 and 17-25 age group are having the highest fatalities and -9 with lowest fatalities.")
st.pyplot(plt)


st.subheader("4) Comparison of Fatalities on Weekends/Nights vs Weekdays/Daytime")
plt.figure(figsize=(10, 8))
sns.countplot(x='Day of week', hue='Time of day', data=df[(df['Time of day'].isin(['Day'])) | (df['Day of week'].isin(['Weekday'])) & (df['Day of week'].isin(['Weekend'])) | (df['Time of day'].isin(['Night']))])
plt.xlabel('Day of week')
plt.ylabel('Count')
plt.title('Comparison of Fatalities on Weekends/Nights vs Weekdays/Daytime')
st.success("The bar graph shows the distribution of road fatalities by Weekends/Weekday vs Night/Day. X-Axis represents the Day of week which are Weekday – Night/Day, Weekend – Night/Day and Y-Axis represents the count fatalities. Moreover, this data is Australian Road fatalities by Day of week – March 2023 data. Each Time of day Night/Day are represented by different colors. Weekday – Day time are having the highest fatalities and – Weekend Day time with lowest fatalities.")
st.pyplot(plt)


st.subheader("5) Distribution of Road Fatalities by Road User and Crash Type")
plt.figure(figsize=(15,10))
sns.countplot(x='Road User', hue='Crash Type', data=df)
plt.xlabel('Road User')
plt.ylabel('Count')
plt.title('Distribution of Road Fatalities by Road User and Crash Type')
st.success("The bar graph shows the distribution of road fatalities by Road User. X-Axis represents the Driver, Motorcycle rider, Passenger, Pedestrian, Pedal cyclist, Motorcycle pillion passenger, Other/-9  and Y-Axis represents the count fatalities. Moreover, this data is Australian Road fatalities by Crash Type – March 2023 data. Each Crash type Road user are represented by different colors. Crash type – Driver (Single, Multiple) are having the highest fatalities and – Crash Type – Others/-9 (Single, Multiple) with lowest fatalities. -9 represents the fields which does not have proper data")
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
report = classification_report(y_test_data,y_prediction)

# Display results
st.subheader("Classification (Predictive Analytics)")
st.success("The target variable is 'Crash Type', while the feature variables are 'Age Group', 'Gender', 'State', 'Time of day', 'Day of the week', and 'Road User'. The categorical feature variables are then converted to numerical variables using one-hot encoding. The data are separated into training and testing collections. The training set comprises 80% of the data, while the assessment set comprises 20%. Finally, the classifier predicts the trial set's target variable to calculate the classifier's accuracy.")
st.write("Accuracy (SVM):", accuracy)

st.text(report)