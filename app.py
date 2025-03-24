import streamlit as st
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import os
from nltk.sentiment import SentimentIntensityAnalyzer
from nlp_utils import extract_keywords, analyze_sentiment, extract_dream_themes
from gpt_predictor import predict_future_impact, analyze_dream_patterns, GOOGLE_API_KEY
from personality import get_personality_data, get_personality_profile, get_dream_processing_style
from visualization import generate_wordcloud, plot_sentiment_over_time, plot_emotion_distribution, plot_theme_correlation, plot_interactive_sentiment_timeline, create_dream_dashboard
from emotion_detection import detect_emotions, analyze_emotion_patterns, get_emotion_recommendations
from dream_symbols import analyze_dream_symbols, get_symbol_frequencies, generate_symbol_insights
import datetime
import numpy as np
import sys

# Version number
APP_VERSION = "1.0.0"

# Initialize NLTK
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

st.set_page_config(page_title="Future Dream Influence Predictor", layout="wide")

# Initialize session state
def init_session_state():
    if 'personality' not in st.session_state:
        st.session_state.personality = {"intuition": 5, "stress": 5, "creativity": 5, "analytical": 5}
    
    if 'dream_history' not in st.session_state:
        try:
            df = pd.read_csv("dream_log.csv")
            # Ensure all required columns exist
            required_columns = ["date", "dream", "themes", "sentiment", "category"]
            for col in required_columns:
                if col not in df.columns:
                    df[col] = "" if col != "sentiment" else 0.0
            
            st.session_state.dream_history = df
        except (FileNotFoundError, pd.errors.EmptyDataError):
            st.session_state.dream_history = pd.DataFrame(
                columns=["date", "dream", "themes", "sentiment", "category", "emotions", "symbols"]
            )

# Call initialization function
init_session_state()

def categorize_dream(themes, sentiment):
    if not themes:
        return "other"
        
    categories = {
        "adventure": ["travel", "explore", "journey", "discover", "adventure"],
        "relationship": ["friend", "family", "love", "partner", "relationship"],
        "fear": ["scary", "afraid", "fear", "terror", "nightmare"],
        "success": ["achieve", "win", "success", "accomplish", "victory"],
        "loss": ["lose", "miss", "gone", "disappear", "loss"]
    }
    
    category_scores = {}
    for category, keywords in categories.items():
        score = sum(1 for theme in themes if any(keyword in theme.lower() for keyword in keywords))
        category_scores[category] = score
    
    if sentiment > 0.3:
        category_scores["success"] = category_scores.get("success", 0) + 1
    elif sentiment < -0.3:
        category_scores["fear"] = category_scores.get("fear", 0) + 1
        category_scores["loss"] = category_scores.get("loss", 0) + 1
    
    max_category = max(category_scores.items(), key=lambda x: x[1]) if category_scores else ("other", 0)
    return max_category[0] if max_category[1] > 0 else "other"

def analyze_patterns(dream_history):
    if not isinstance(dream_history, pd.DataFrame) or len(dream_history) < 2:
        return "Not enough dream data to analyze patterns."
    
    try:
        sentiment_trend = "increasing" if dream_history["sentiment"].iloc[-1] > dream_history["sentiment"].mean() else "decreasing"
        
        all_themes = []
        for themes_str in dream_history["themes"]:
            if isinstance(themes_str, str):
                all_themes.extend([t.strip() for t in themes_str.split(",")])
        
        from collections import Counter
        common_themes = Counter(all_themes).most_common(3) if all_themes else [("unknown", 0)]
        
        if "category" in dream_history.columns and not dream_history["category"].empty:
            most_common_category = dream_history["category"].mode()[0]
        else:
            most_common_category = "unknown"
        
        analysis = f"Your dream sentiment is {sentiment_trend}. "
        analysis += f"Your most common dream themes are: {', '.join([t[0] for t in common_themes])}. "
        analysis += f"Your dreams most frequently fall into the '{most_common_category}' category."
        
        return analysis
    except Exception as e:
        return f"Error analyzing patterns: {str(e)}"

def generate_recommendations(dream_history, personality):
    if not isinstance(dream_history, pd.DataFrame) or len(dream_history) < 2:
        return "Not enough dream data to generate personalized recommendations."
    
    try:
        recommendations = []
        
        recent_sentiments = dream_history["sentiment"].tail(3)
        if recent_sentiments.mean() < -0.2:
            recommendations.append("Your recent dreams show negative emotions. Consider stress-reduction activities like meditation or exercise.")
        
        all_themes = []
        for themes_str in dream_history["themes"].tail(5):
            if isinstance(themes_str, str):
                all_themes.extend([t.strip() for t in themes_str.split(",")])
        
        from collections import Counter
        theme_counts = Counter(all_themes)
        recurring_themes = [theme for theme, count in theme_counts.items() if count > 1]
        
        if recurring_themes:
            recommendations.append(f"You have recurring themes of {', '.join(recurring_themes[:3])}. These may represent unresolved issues or important aspects of your life to focus on.")
        
        if personality["intuition"] > 7:
            recommendations.append("With your high intuition score, pay special attention to symbolic elements in your dreams as they may provide insights into decisions you're facing.")
        
        if personality["stress"] > 7:
            recommendations.append("Your high stress level may be influencing your dream patterns. Consider journaling before bed to process daily thoughts.")
        
        if personality["creativity"] > 7:
            recommendations.append("Channel the creative energy from your dreams into artistic expression - this may help process and integrate dream insights.")
        
        if personality["analytical"] > 7:
            recommendations.append("Try analyzing patterns between your daily activities and dream content to identify potential correlations.")
        
        if not recommendations:
            recommendations.append("Continue tracking your dreams to reveal more patterns and insights over time.")
        
        return "\n\n".join(recommendations)
    except Exception as e:
        return f"Error generating recommendations: {str(e)}"

def save_dream_history():
    try:
        if st.session_state.dream_history is not None:
            st.session_state.dream_history.to_csv("dream_log.csv", index=False)
            return True
    except Exception as e:
        st.error(f"Error saving dream history: {str(e)}")
        return False
    return False

# Navigation sidebar
page = st.sidebar.radio("Go to", ["Dream Input", "Dream History", "Analysis & Insights", "Settings"])

# App version in sidebar
st.sidebar.markdown(f"<div style='text-align: center; color: gray; font-size: small;'>Version {APP_VERSION}</div>", unsafe_allow_html=True)

if page == "Dream Input":
    st.title("Future Dream Influence Predictor")
    st.write("Enter your dream details below to analyze potential future influences.")
    
    dream_text = st.text_area("Describe your dream:", height=150)
    dream_date = st.date_input("Dream date:", datetime.date.today())
    
    if st.button("Analyze Dream"):
        if dream_text:
            with st.spinner("Analyzing your dream..."):
                try:
                    themes = extract_keywords(dream_text)
                    sentiment_scores = analyze_sentiment(dream_text)
                    compound_sentiment = sentiment_scores.get('compound', 0)
                    dream_themes = extract_dream_themes(dream_text)
                    category = categorize_dream(themes, compound_sentiment)
                    
                    emotions = detect_emotions(dream_text)
                    primary_emotions = emotions['primary_emotions']
                    emotion_scores = emotions['emotion_scores']
                    
                    symbol_analysis = analyze_dream_symbols(dream_text, st.session_state.personality)
                    symbols_found = symbol_analysis['symbols_found']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Dream Themes")
                        st.write(", ".join(themes))
                        st.write("Higher-Level Themes: ", ", ".join(dream_themes))
                        
                        st.subheader("Sentiment Analysis")
                        st.write(f"Compound Score: {compound_sentiment:.2f}")
                        st.write(f"Positive: {sentiment_scores.get('pos', 0):.2f}")
                        st.write(f"Neutral: {sentiment_scores.get('neu', 0):.2f}")
                        st.write(f"Negative: {sentiment_scores.get('neg', 0):.2f}")
                        
                        st.subheader("Dream Category")
                        st.write(category.capitalize())
                        
                        st.subheader("Emotions Detected")
                        if primary_emotions:
                            st.write(", ".join(primary_emotions))
                        else:
                            st.write("Neutral")
                    
                    with col2:
                        st.subheader("Word Cloud of Themes")
                        if themes:
                            fig = generate_wordcloud(themes)
                            st.pyplot(fig)
                        else:
                            st.write("Not enough themes to generate word cloud.")
                        
                        if symbols_found:
                            st.subheader("Dream Symbols")
                            st.write(", ".join(symbols_found[:5]))
                            with st.expander("Symbol Interpretation"):
                                st.write(symbol_analysis['interpretation'])
                    
                    st.subheader("Future Influence Prediction")
                    prediction = predict_future_impact(dream_themes, compound_sentiment, st.session_state.personality, 
                                                    dream_text=dream_text, emotions=emotion_scores, symbols=symbols_found)
                    st.write(prediction)
                    
                    st.subheader("Personalized Recommendations")
                    tabs = st.tabs(["Emotion-Based", "Symbol-Based"])
                    with tabs[0]:
                        recommendations = get_emotion_recommendations(emotion_scores, st.session_state.personality)
                        st.write(recommendations)
                    with tabs[1]:
                        st.write(symbol_analysis['recommendations'])
                    
                    emotions_str = ", ".join(primary_emotions) if primary_emotions else "neutral"
                    symbols_str = ", ".join(symbols_found) if symbols_found else ""
                    
                    new_entry = {
                        "date": dream_date.strftime("%Y-%m-%d"),
                        "dream": dream_text,
                        "themes": ", ".join(themes),
                        "sentiment": compound_sentiment,
                        "category": category,
                        "emotions": emotions_str,
                        "symbols": symbols_str
                    }
                    
                    # Add new dream entry
                    st.session_state.dream_history = pd.concat([st.session_state.dream_history, 
                                                            pd.DataFrame([new_entry])], 
                                                            ignore_index=True)
                    
                    # Save to file
                    if save_dream_history():
                        st.success("Dream analyzed and saved to history!")
                except Exception as e:
                    st.error(f"Error analyzing dream: {str(e)}")
        else:
            st.error("Please enter a dream description.")

elif page == "Dream History":
    st.title("Dream History")
    
    if isinstance(st.session_state.dream_history, pd.DataFrame) and len(st.session_state.dream_history) > 0:
        st.write(f"You have recorded {len(st.session_state.dream_history)} dreams.")
        
        st.subheader("Filter Dreams")
        col1, col2 = st.columns(2)
        with col1:
            date_range = st.date_input("Date range", 
                                      [datetime.date.today() - datetime.timedelta(days=30), datetime.date.today()])
        with col2:
            if "category" in st.session_state.dream_history.columns:
                all_categories = st.session_state.dream_history["category"].dropna().unique()
                categories = ["All"] + list(all_categories)
                selected_category = st.selectbox("Category", categories)
        
        try:
            filtered_df = st.session_state.dream_history.copy()
            if len(date_range) == 2:
                filtered_df = filtered_df[(filtered_df["date"] >= date_range[0].strftime("%Y-%m-%d")) & 
                                        (filtered_df["date"] <= date_range[1].strftime("%Y-%m-%d"))]
            
            if "category" in filtered_df.columns and selected_category != "All":
                filtered_df = filtered_df[filtered_df["category"] == selected_category]
            
            st.subheader("Dream Records")
            display_columns = ["date", "dream", "themes", "sentiment", "category"]
            # Only include columns that exist
            display_columns = [col for col in display_columns if col in filtered_df.columns]
            st.dataframe(filtered_df[display_columns])
            
            if len(filtered_df) > 1:
                st.subheader("Sentiment Over Time")
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    filtered_df["date"] = pd.to_datetime(filtered_df["date"])
                    plot_sentiment_over_time(filtered_df)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error plotting sentiment: {str(e)}")
                
                if "category" in filtered_df.columns:
                    try:
                        st.subheader("Dream Category Distribution")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        category_counts = filtered_df["category"].value_counts()
                        category_counts.plot(kind="bar", ax=ax)
                        plt.title("Dream Categories")
                        plt.ylabel("Count")
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error plotting categories: {str(e)}")
        except Exception as e:
            st.error(f"Error processing dream history: {str(e)}")
    else:
        st.info("No dream records found. Add dreams on the Dream Input page.")

elif page == "Analysis & Insights":
    st.title("Dream Analysis & Insights")
    
    if isinstance(st.session_state.dream_history, pd.DataFrame) and len(st.session_state.dream_history) > 1:
        st.subheader("Dream Patterns")
        if len(st.session_state.dream_history) >= 3:
            pattern_analysis = analyze_dream_patterns(st.session_state.dream_history, st.session_state.personality)
            st.write(pattern_analysis)
        else:
            pattern_analysis = analyze_patterns(st.session_state.dream_history)
            st.write(pattern_analysis)
        
        st.subheader("Personalized Recommendations")
        recommendations = generate_recommendations(st.session_state.dream_history, st.session_state.personality)
        st.write(recommendations)
        
        st.subheader("Dream Insights Visualization")
        
        tabs = st.tabs(["Sentiment Analysis", "Emotion Analysis", "Theme Analysis", "Dashboard"])
        
        with tabs[0]:
            st.subheader("Distribution of Dream Sentiment")
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.hist(st.session_state.dream_history["sentiment"], bins=10, alpha=0.7)
                plt.title("Distribution of Dream Sentiment")
                plt.xlabel("Sentiment Score")
                plt.ylabel("Frequency")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error creating sentiment histogram: {str(e)}")
            
            st.subheader("Interactive Sentiment Timeline")
            try:
                interactive_fig = plot_interactive_sentiment_timeline(st.session_state.dream_history)
                if interactive_fig:
                    st.plotly_chart(interactive_fig, use_container_width=True)
                else:
                    st.info("Not enough data to create interactive timeline.")
            except Exception as e:
                st.error(f"Error creating sentiment timeline: {str(e)}")
        
        with tabs[1]:
            if 'emotions' in st.session_state.dream_history.columns:
                st.subheader("Emotion Distribution")
                try:
                    emotion_fig = plot_emotion_distribution(st.session_state.dream_history)
                    if emotion_fig:
                        st.pyplot(emotion_fig)
                    else:
                        st.info("Not enough emotion data for visualization.")
                except Exception as e:
                    st.error(f"Error plotting emotions: {str(e)}")
                    
                if len(st.session_state.dream_history) >= 3:
                    st.subheader("Emotion Pattern Analysis")
                    try:
                        emotion_analysis = analyze_emotion_patterns(st.session_state.dream_history)
                        st.write(emotion_analysis)
                    except Exception as e:
                        st.error(f"Error analyzing emotion patterns: {str(e)}")
            else:
                st.info("No emotion data available. Add more dreams with emotion analysis to see insights here.")
        
        with tabs[2]:
            # Theme correlation
            if len(st.session_state.dream_history) >= 5:
                st.subheader("Theme Correlation Analysis")
                try:
                    theme_fig = plot_theme_correlation(st.session_state.dream_history)
                    if theme_fig:
                        st.pyplot(theme_fig)
                    else:
                        st.info("Not enough theme data for correlation analysis.")
                except Exception as e:
                    st.error(f"Error creating theme correlation: {str(e)}")
                    
                # Symbol analysis
                if 'symbols' in st.session_state.dream_history.columns:
                    st.subheader("Dream Symbol Insights")
                    try:
                        symbol_insights = generate_symbol_insights(st.session_state.dream_history, st.session_state.personality)
                        st.write(symbol_insights)
                    except Exception as e:
                        st.error(f"Error generating symbol insights: {str(e)}")
            else:
                st.info("Need at least 5 dream records for theme correlation analysis. Please add more dreams.")
        
        with tabs[3]:
            # Comprehensive dashboard
            if len(st.session_state.dream_history) >= 3:
                st.subheader("Dream Analysis Dashboard")
                try:
                    dashboard = create_dream_dashboard(st.session_state.dream_history)
                    if dashboard:
                        st.plotly_chart(dashboard, use_container_width=True)
                    else:
                        st.info("Could not create dashboard with available data.")
                except Exception as e:
                    st.error(f"Error creating dashboard: {str(e)}")
            else:
                st.info("Need at least 3 dream records for the dashboard. Please add more dreams.")
        
        # Theme frequency over time
        if len(st.session_state.dream_history) >= 5:
            st.subheader("Theme Evolution Over Time")
            
            try:
                # Extract all unique themes
                all_themes = set()
                for themes_str in st.session_state.dream_history["themes"]:
                    if isinstance(themes_str, str):
                        all_themes.update([t.strip() for t in themes_str.split(",")])
                
                if all_themes:
                    # Get top 5 themes
                    theme_counts = {}
                    for theme in all_themes:
                        count = sum(1 for themes_str in st.session_state.dream_history["themes"] 
                                  if isinstance(themes_str, str) and theme in themes_str)
                        theme_counts[theme] = count
                    
                    top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                    top_theme_names = [t[0] for t in top_themes]
                    
                    # Create dataframe for theme evolution
                    theme_evolution = pd.DataFrame()
                    theme_evolution["date"] = st.session_state.dream_history["date"]
                    
                    for theme in top_theme_names:
                        theme_evolution[theme] = st.session_state.dream_history["themes"].apply(
                            lambda x: 1 if isinstance(x, str) and theme in x else 0
                        )
                    
                    # Plot theme evolution
                    theme_evolution["date"] = pd.to_datetime(theme_evolution["date"])
                    theme_evolution = theme_evolution.sort_values("date")
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    for theme in top_theme_names:
                        plt.plot(theme_evolution["date"], 
                                theme_evolution[theme].rolling(window=min(3, len(theme_evolution)), min_periods=1).mean(), 
                                marker='o', label=theme)
                    
                    plt.title("Theme Frequency Over Time (3-dream rolling average)")
                    plt.xlabel("Date")
                    plt.ylabel("Frequency")
                    plt.legend()
                    st.pyplot(fig)
                else:
                    st.info("No theme data available for visualization.")
            except Exception as e:
                st.error(f"Error plotting theme evolution: {str(e)}")
    else:
        st.info("You need at least 2 dream records for analysis. Please add more dreams on the Dream Input page.")

# Settings Page
elif page == "Settings":
    st.title("Settings")
    
    st.subheader("Personality Profile")
    st.write("Adjust your personality traits to improve prediction accuracy. Rate each trait on a scale from 1 (low) to 10 (high).")
    
    col1, col2 = st.columns(2)
    
    with col1:
        intuition = st.slider("Intuition-driven decision making", 1, 10, st.session_state.personality["intuition"])
        stress = st.slider("Typical stress level", 1, 10, st.session_state.personality["stress"])
    
    with col2:
        creativity = st.slider("Creative thinking", 1, 10, st.session_state.personality["creativity"])
        analytical = st.slider("Analytical thinking", 1, 10, st.session_state.personality["analytical"])
    
    if st.button("Save Settings"):
        st.session_state.personality = {
            "intuition": intuition,
            "stress": stress,
            "creativity": creativity,
            "analytical": analytical
        }
        st.success("Settings saved successfully!")
    
    st.subheader("Data Management")
    if st.button("Clear Dream History"):
        if isinstance(st.session_state.dream_history, pd.DataFrame) and len(st.session_state.dream_history) > 0:
            confirm = st.checkbox("Are you sure? This action cannot be undone.")
            if confirm:
                st.session_state.dream_history = pd.DataFrame(
                    columns=["date", "dream", "themes", "sentiment", "category", "emotions", "symbols"]
                )
                try:
                    save_dream_history()
                    st.success("Dream history cleared successfully!")
                except Exception as e:
                    st.error(f"Error clearing dream history: {str(e)}")
        else:
            st.info("No dream history to clear.")

    st.subheader("About")
    st.write("""
    **Future Dream Influence Predictor**
    
    This application analyzes your dreams and predicts how they might influence your future decisions.
    It uses natural language processing and sentiment analysis to extract themes and emotions from your dream descriptions.
    
    The more dreams you record, the more accurate the pattern analysis and predictions will become.
    """)
    
    st.write(f"**Version**: {APP_VERSION}")
    
    # API Key configuration for GPT predictor
    st.subheader("API Configuration")
    api_placeholder = "Your API key is configured in gpt_predictor.py"
    if GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
        api_status = "Not configured - Prediction features disabled"
        st.warning("API key not configured. Prediction features are disabled.")
    else:
        masked_key = "•" * 8 + GOOGLE_API_KEY[-4:] if len(GOOGLE_API_KEY) > 4 else "•" * 4
        api_status = f"Configured {masked_key}"
    
    st.text_input("Google AI API Key Status", value=api_status, disabled=True, 
                help="To set your API key, edit the GOOGLE_API_KEY variable in gpt_predictor.py")
    
    st.info("To use the generative AI features, you need to set your Google Gemini API key in the gpt_predictor.py file. Get an API key from https://makersuite.google.com/app/apikey")