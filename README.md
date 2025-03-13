# Future Dream Influence Predictor

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](license) 

## Overview

![Response](https://raw.githubusercontent.com/dhrrishit/Future-Dream-Influence-Predictor/main/Work.jpg)

The **Future Dream Influence Predictor** is a Python-based application that analyzes your dreams to extract key themes, assess sentiment, and creatively predict how these dream elements might influence your future decisions. Leveraging Natural Language Processing (NLP) with spaCy and NLTK, sentiment analysis with VADER, and the powerful Gemini Pro model from Google AI, this project offers a fun and insightful look into the potential impact of your subconscious.

**Imagine:** You've just had a vivid dream. You describe it to the predictor, and it not only tells you the dominant themes and the overall feeling of the dream but also gives you a creative glimpse into how these dream elements might play out in your waking life.

**This project features:**

* **Dream Theme Extraction:** Uses spaCy to identify and extract significant keywords (nouns, verbs, adjectives) from your dream description.
* **Enhanced Emotion Detection:** Utilizes a comprehensive emotion lexicon and intensity modifiers to provide detailed emotional analysis of your dreams.
* **Advanced Dream Symbol Analysis:** Interprets 25+ common dream symbols with detailed psychological interpretations.
* **Sentiment Analysis:** Employs NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) to analyze the sentiment expressed in your dream text, providing scores for positive, negative, neutral, and compound sentiment.
* **Future Influence Prediction:** Integrates with Google's Gemini Pro model to generate creative and thoughtful predictions about how your dream themes and sentiment could influence your future decisions, taking into account your personality traits.
* **Interactive Personality Input:** Gathers personality data (intuition, stress level, creativity, and analytical thinking) through a simple interactive questionnaire to personalize the future influence prediction.
* **Pattern Analysis:** Identifies recurring dream themes and emotions over time, providing insights into your subconscious patterns.
* **Category-based Classification:** Organizes dreams into meaningful categories for better understanding and analysis.
* **Enhanced Visualizations:**
    * **Interactive Sentiment Timeline:** Shows sentiment changes over time with rolling averages.
    * **Emotion Distribution Plots:** Visualizes the distribution and intensity of detected emotions.
    * **Theme Correlation Analysis:** Reveals relationships between different dream themes.
    * **Symbol Frequency Visualization:** Tracks the occurrence of common dream symbols.
    * **Word Cloud:** Generates a visually appealing word cloud from the extracted dream themes.
    * **Comprehensive Dream Dashboard:** Combines multiple visualizations into a single interactive interface.
* **Dream Logging:** Saves your dream descriptions, extracted themes, sentiment scores, emotions, symbols, and dates to a CSV file (`dream_log.csv`), enabling temporal analysis and trend tracking.

## Key Features

* **Advanced NLP Analysis:** Utilizes spaCy and NLTK for robust dream text processing, including enhanced emotion detection and comprehensive dream symbol analysis.
* **Gemini Pro Integration:** Leverages Google's advanced language model for creative future predictions.
* **Personalized Insights:** Provides emotion-based and symbol-based recommendations tailored to your personality traits and recurring patterns.
* **Rich Visualizations:** Offers interactive timelines, distribution plots, correlation analyses, and symbol tracking for comprehensive dream data understanding.
* **Simple and Interactive:** Features both a command-line interface and a Streamlit web application for dream input and results visualization.

## Getting Started

Follow these steps to get the Future Dream Influence Predictor running on your local machine.

### Prerequisites

* **Python 3.7 or higher:** Make sure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).
* **pip:** Python package installer (usually comes with Python).
* **Google Gemini API Key:** You will need an API key from Google AI Studio to use the Gemini Pro model. Get your API key [here](https://makersuite.google.com/app/apikey).
* **spaCy English Language Model:** The project uses `en_core_web_sm` from spaCy.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dhrrishit/Future-Dream-Influence-Predictor.git
   cd Future-Dream-Influence-Predictor
   ```

2. **Create a virtual environment (recommended):**
    ```python -m venv venv
     source venv/bin/activate  # On Linux/macOS
     venv\Scripts\activate  # On Windows
    ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

5. **Download spaCy English model:**
   ```
   python -m spacy download en_core_web_sm
   ```

6. **Set up your Google Gemini API Key:**

-    Open the gpt_predictor.py file.
-    Replace "YOUR_API_KEY_HERE" with your actual Gemini API key in the line:
  
   ```GOOGLE_API_KEY = "YOUR_API_KEY_HERE"```


## Running the Predictor

1. **Run the application:**

   **For command-line interface:**
   ```
   python main.py
   ```
   
   **For web application:**
   ```
   streamlit run app.py
   ```

2. **Follow the prompts:**
   - You will be asked to enter your dream description. Be as detailed as you like!
   - You will then be prompted to answer questions about your personality (intuition, stress level, creativity, and analytical thinking) on a scale of 1 to 10.
   - The application will analyze your dream using enhanced emotion detection and comprehensive symbol analysis.

3. **View the results:**
   - Extracted Dream Themes (as a list)
   - Dream Sentiment Scores (from VADER)
   - Detailed Emotion Analysis with intensity levels
   - Dream Symbol Interpretations (25+ common symbols)
   - Interactive Visualizations:
     * Sentiment Timeline with rolling averages
     * Emotion Distribution Plots
     * Theme Correlation Analysis
     * Symbol Frequency Charts
   - Pattern Analysis of recurring themes and emotions
   - Future Influence Prediction generated by Gemini Pro
   - Personalized recommendations based on emotions and symbols
   - Your dream entry will be saved to dream_log.csv with temporal data

```The script will output:
Extracted Dream Themes (as a list)
Dream Sentiment Scores (from VADER)
Emotions Detected in the dream
Dream Symbols identified
A Word Cloud visualization will be displayed.
Future Influence Prediction generated by Gemini Pro.
Personalized recommendations based on emotions and symbols.
If you have previous dream logs in dream_log.csv, visualizations of patterns over time will also be displayed.
Your dream entry will be saved to dream_log.csv.
```

## Dependencies

- google-generativeai: For interacting with Google Gemini models.
- spacy: For advanced Natural Language Processing tasks (tokenization, lemmatization, POS tagging).
- en-core-web-sm: spaCy's English language model.
- nltk: Natural Language Toolkit for sentiment analysis (VADER) and other NLP tasks.
- pandas: For data manipulation and working with dream logs in CSV format.
- matplotlib: For plotting sentiment over time.
- seaborn: Built on top of matplotlib, for enhanced data visualizations (sentiment plots).
- wordcloud: For generating word cloud visualizations of dream themes.
- plotly: For creating interactive visualizations and dashboards.
- streamlit: For creating the web application interface.

## Example of Future Dream Influence Predictor

Enter your dream description: I dreamed I was flying over a lush green forest, but suddenly a storm appeared and I started falling. I felt scared and lost.
... (Word cloud visualization will appear) ...
Please answer the following questions on a scale from 1 (low) to 10 (high):
How much do you rely on intuition for decision-making? 7
How high is your typical stress level? 6
How would you rate your creativity? 8
How would you rate your analytical thinking? 5

Extracted Dream Themes: ['lush', 'fly', 'green', 'forest', 'dream', 'appear', 'storm', 'start', 'fall', 'lose', 'scare']
Dream Sentiment Scores: {'neg': 0.259, 'neu': 0.538, 'pos': 0.203, 'compound': -0.1779}
Emotions Detected: fear, sadness
Dream Symbols: flying, forest, storm, falling

- **Future Influence Prediction**
Based on your dream, the themes of flying, lush greenery, and a sudden storm, combined with a slightly negative sentiment and moderate intuition and stress levels, here's a potential future influence prediction:

Your dream may reflect a current period of ambition and exploration (flying over a lush forest), but also underlying anxieties or potential challenges on the horizon (the sudden storm and falling). Your intuition score suggests you are somewhat guided by your gut feelings, which might be heightened by the dream's unsettling elements. The moderate stress level indicates you are likely equipped to handle challenges, but the dream could be a signal to prepare for unexpected turbulence in your plans.

Future decisions might be influenced by a cautious optimism. You may be drawn to opportunities that initially seem promising and expansive, but you might also be more aware of potential risks and setbacks. This dream could encourage you to plan carefully, consider contingency plans, and not be overly swayed by initial excitement without assessing potential storms ahead. Pay attention to your intuition but also balance it with practical considerations as you navigate upcoming decisions.

## Contributing

*Contributions are welcome! If you have ideas for improvements, bug fixes, or new features, please feel free to:*

- ```Fork the repository```
- ```Create a new branch for your feature (git checkout -b feature/your-feature).```
- ```Commit your changes (git commit -am 'Add some feature').```
- ```Push to the branch (git push origin feature/your-feature).```
- ```Open a Pull Request.```

## Author
**Dhrrishit Deka**
- **Email**: dhrrishit@gmail.com
- **GitHub**: https://github.com/dhrrishit
- **LinkedIn**: https://www.linkedin.com/in/dhrrishitdeka/
- **X (Twitter)**: https://x.com/dhrrishitdeka
