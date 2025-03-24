import sys
import os

# Initialize NLTK
print("Initializing NLTK...")
try:
    import nltk
    nltk.download('vader_lexicon', quiet=True)
except ImportError:
    print("Error: NLTK not found. Please install with: pip install nltk")
    sys.exit(1)
except Exception as e:
    print(f"Error initializing NLTK: {e}")
    print("Some functionality might be limited.")

# Import required packages
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    from nlp_utils import extract_keywords, analyze_sentiment, extract_dream_themes
    from gpt_predictor import predict_future_impact, analyze_dream_patterns, initialize_model, GOOGLE_API_KEY
    from personality import get_personality_data, get_personality_profile, get_dream_processing_style
    from visualization import generate_wordcloud, plot_sentiment_over_time, plot_emotion_distribution
    from emotion_detection import detect_emotions, analyze_emotion_patterns, get_emotion_recommendations
    from dream_symbols import analyze_dream_symbols, generate_symbol_insights
except ImportError as e:
    print(f"Error: Required module not found: {e}")
    print("Please install required dependencies using: pip install -r requirements.txt")
    sys.exit(1)

def main():
    """Main function for the command-line interface of the Future Dream Influence Predictor."""
    print("=== Future Dream Influence Predictor ===\n")
    
    # Check API key for GPT features
    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
        print("Warning: API key not configured.")
        print("Generative AI features will be disabled.")
        print("To enable these features, set your API key in gpt_predictor.py\n")
    else:
        # Initialize model if API key is available
        initialize_model()

    try:
        dream_text = input("Enter your dream description: ")
        
        if not dream_text.strip():
            print("Error: Empty dream description. Please provide details about your dream.")
            return

        print("\nAnalyzing your dream...")
        themes = extract_keywords(dream_text)
        sentiment_scores = analyze_sentiment(dream_text)
        compound_sentiment = sentiment_scores.get('compound', 0)
        dream_themes = extract_dream_themes(dream_text)

        emotions = detect_emotions(dream_text)
        primary_emotions = emotions['primary_emotions']
        emotion_scores = emotions['emotion_scores']

        symbol_analysis = analyze_dream_symbols(dream_text)
        symbols_found = symbol_analysis['symbols_found']

        print("\n=== Dream Analysis ===")
        print("\nExtracted Dream Themes:", ", ".join(themes) if themes else "No specific themes detected")
        print("Higher-Level Dream Themes:", ", ".join(dream_themes) if dream_themes else "No specific themes detected")
        print("\nDream Sentiment:")
        print(f"  Compound Score: {compound_sentiment:.2f}")
        print(f"  Positive: {sentiment_scores.get('pos', 0):.2f}")
        print(f"  Neutral: {sentiment_scores.get('neu', 0):.2f}")
        print(f"  Negative: {sentiment_scores.get('neg', 0):.2f}")
        
        print("\nPrimary Emotions Detected:", ", ".join(primary_emotions) if primary_emotions else "Neutral")
        
        if symbols_found:
            print("\nSignificant Dream Symbols:", ", ".join(symbols_found[:5]))
            print("\nSymbol Interpretation:")
            print(symbol_analysis['interpretation'])

        print("\nGenerating word cloud of dream themes...")
        try:
            if themes:
                fig = generate_wordcloud(themes)
                plt.show()
            else:
                print("Not enough themes to generate word cloud.")
        except Exception as e:
            print(f"Error generating word cloud: {e}")

        print("\n=== Personality Assessment ===\n")
        try:
            personality = get_personality_data()
            
            personality_profile = get_personality_profile(personality)
            print("\n" + personality_profile)
            
            processing_style = get_dream_processing_style(personality)
            print("\n" + processing_style)
        except Exception as e:
            print(f"Error during personality assessment: {e}")
            personality = {"intuition": 5, "stress": 5, "creativity": 5, "analytical": 5}

        print("\n=== Future Influence Prediction ===\n")
        try:
            if GOOGLE_API_KEY != "YOUR_API_KEY_HERE":
                prediction = predict_future_impact(dream_themes, compound_sentiment, personality, 
                                            dream_text=dream_text, emotions=emotion_scores, 
                                            symbols=symbols_found)
                print(prediction)
            else:
                print("Future prediction is disabled. Configure your API key in gpt_predictor.py")
        except Exception as e:
            print(f"Error generating prediction: {e}")

        print("\n=== Personalized Recommendations ===\n")
        try:
            recommendations = get_emotion_recommendations(emotion_scores, personality)
            print(recommendations)
            
            print("\n=== Symbol-Based Recommendations ===\n")
            print(symbol_analysis['recommendations'])
        except Exception as e:
            print(f"Error generating recommendations: {e}")

        # Save dream to history
        try:
            log_entry = {"date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "dream": dream_text,
                        "themes": ", ".join(themes) if themes else "",
                        "sentiment": compound_sentiment,
                        "emotions": emotions['emotions_str'],
                        "symbols": ", ".join(symbols_found) if symbols_found else ""}
            try:
                dream_log = pd.read_csv("dream_log.csv")
                dream_log = pd.concat([dream_log, pd.DataFrame([log_entry])], ignore_index=True)
            except (FileNotFoundError, pd.errors.EmptyDataError):
                dream_log = pd.DataFrame([log_entry])
            
            dream_log.to_csv("dream_log.csv", index=False)
            print("\nDream saved to history log.")
        except Exception as e:
            print(f"Error saving dream to history: {e}")

        # Historical analysis
        try:
            if os.path.exists("dream_log.csv"):
                dream_log = pd.read_csv("dream_log.csv")
                if len(dream_log) > 1:
                    print("\n=== Dream History Analysis ===\n")
                    print("Plotting sentiment over time...")
                    fig = plot_sentiment_over_time(dream_log)
                    plt.show()
                    
                    if len(dream_log) >= 3:
                        try:
                            if GOOGLE_API_KEY != "YOUR_API_KEY_HERE":
                                pattern_analysis = analyze_dream_patterns(dream_log, personality)
                                print("\nDream Pattern Analysis:")
                                print(pattern_analysis)
                            else:
                                print("\nDream pattern analysis requires API key configuration.")
                        except Exception as e:
                            print(f"Error analyzing dream patterns: {e}")
                        
                        if 'emotions' in dream_log.columns:
                            try:
                                emotion_analysis = analyze_emotion_patterns(dream_log)
                                print("\nEmotion Pattern Analysis:")
                                print(emotion_analysis)
                                
                                fig = plot_emotion_distribution(dream_log)
                                if fig:
                                    plt.show()
                            except Exception as e:
                                print(f"Error analyzing emotion patterns: {e}")
        except Exception as e:
            print(f"Error analyzing dream history: {e}")
        
        print("\nAnalysis complete.")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please try again or check the application files.")

if __name__ == "__main__":
    main()