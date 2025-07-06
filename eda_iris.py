import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset from sklearn
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(df.head())

   # inspected the data set

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Data Types and Nulls ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

    # Visualize and Interpret

sns.countplot(x='species', data=df)
plt.title("Count of Each Iris Species")
plt.show()

sns.pairplot(df, hue='species')
plt.suptitle("Pairplot of All Features by Species", y=1.02)
plt.show()

sns.boxplot(x='species', y='petal length (cm)', data=df)
plt.title("Petal Length by Species")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.drop(columns='species').corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Features")
plt.show()

print("\n--- Key Insights ---")
print("1. Setosa is clearly distinct in petal length and width — very small values.")
print("2. Petal length and width are highly correlated — useful for classification.")
print("3. Versicolor and Virginica have overlapping features, but separable.")

