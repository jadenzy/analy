# Install necessary libraries
# pip install pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 1: Load the Data
data = pd.read_csv('fake_student_records.csv')

# Step 2: Explore the DataX
print(data.head())      # Preview the first few rows
print(data.info())      # Get a summary of the dataset
print(data.describe())  # Get descriptive statistics

# Step 3: Clean the Data
data = data.dropna()  # Drop rows with missing values
data['column_name'] = data['column_name'].fillna(data['column_name'].mean())  # Fill missing values

# Step 4: Analyze the Data
grouped_data = data.groupby('category_column').mean()
print(grouped_data)

# Step 5: Visualize the Data
sns.barplot(x='category_column', y='value_column', data=data)
plt.show()

# Step 6: Run Advanced Analysis (Linear Regression Example)
X = data[['feature1', 'feature2']]  # Features
y = data['target']                  # Target variable

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
print(predictions)


