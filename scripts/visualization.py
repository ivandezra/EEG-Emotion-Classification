import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.show()

def plot_emotion_distribution(y, labels):
    emotion_counts = y.value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(emotion_counts, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Distribution of Emotions")
    plt.axis('equal')
    plt.show()
