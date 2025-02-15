import nltk
nltk.download('vader_lexicon')
import pandas as pd
from nlp_utils import extract_keywords, analyze_sentiment
from gpt_predictor import predict_future_impact
from personality import get_personality_data
from visualization import generate_wordcloud, plot_sentiment_over_time

def main():
    print("=== Future Influence Predictor ===")

    dream_text = input("Enter your dream description: ")

    themes = extract_keywords(dream_text)
    sentiment_scores = analyze_sentiment(dream_text)
    compound_sentiment = sentiment_scores.get('compound', 0)

    print("\nExtracted Dream Themes:", themes)
    print("Dream Sentiment Scores:", sentiment_scores)

    generate_wordcloud(themes)
    personality = get_personality_data()
    prediction = predict_future_impact(themes, compound_sentiment, personality)
    print("\n--- Future Influence Prediction ---")
    print(prediction)

    log_entry = {"date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "dream": dream_text,
                 "themes": ", ".join(themes),
                 "sentiment": compound_sentiment}
    try:
        dream_log = pd.read_csv("dream_log.csv")
        dream_log = dream_log.append(log_entry, ignore_index=True)
    except FileNotFoundError:
        dream_log = pd.DataFrame([log_entry])
    dream_log.to_csv("dream_log.csv", index=False)

    if len(dream_log) > 1:
        plot_sentiment_over_time(dream_log)

if __name__ == "__main__":
    main()