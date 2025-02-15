import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def plot_sentiment_over_time(dream_log):
    sns.lineplot(data=dream_log, x="date", y="sentiment", marker="o")
    plt.title("Dream Sentiment Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sentiment (Compound Score)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def generate_wordcloud(theme_list):
    all_words = " ".join(theme_list)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()