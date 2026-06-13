import matplotlib.pyplot as plt
import seaborn as sns

print(df.shape)
print(df.info())
print(df.isnull().mean() * 100)

# 타겟 분포
sns.countplot(data=df, x="Churn")
plt.title("Churn Distribution")
plt.show()

# 수치형 변수 분포
df[num_cols].hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# Boxplot
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("MonthlyCharges by Churn")
plt.show()

# 상관관계 Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df[num_cols + ["Churn"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 범주형 변수 Barplot
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Contract Type and Churn")
plt.show()
