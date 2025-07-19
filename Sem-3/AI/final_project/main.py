import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Load your dataset
file_path = r"E:\Sem3-GH\Sem3Lab\AI\final_project\Video_Games.csv"
df = pd.read_csv(file_path)
# Check the dataset and handle missing values if any
print(df.head()) # Check the structure of the dataset
print(df.isnull().sum()) # Check for missing values
# Handling missing values - replace 'tbd' in 'User_Score' with NaN
df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')
# Assuming 'Genre' is the target variable
x = df[['Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']]
y = df['Genre']
# Drop rows with NaN values in the features
x.dropna(inplace=True)
y = y.iloc[x.index] # Update y based on the filtered x indices
# Applying KNN
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print(f"KNN on data accuracy: {accuracy}")
# Applying KMeans clustering
numeric_columns = ['Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']
x = df[numeric_columns].dropna() # Drop rows with NaN values
# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)
# KMeans clustering
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
kmeans.fit(X_scaled)
# Plotting clusters
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis', edgecolor='k', label='Data Points')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='*', s=100, c='red', label='Cluster Centers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
# Adjust x-axis and y-axis limits to show more data points
plt.xlim(X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5)
plt.ylim(X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5)
plt.title('K-means Clustering on Video Games Dataset')
plt.legend()
plt.show()