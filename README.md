# Future Dream Influence Predictor

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](license) 

## Overview


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
* **Improved Error Handling:** Robust error handling throughout the application to ensure stable operation even with missing dependencies or incomplete data.

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
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3. **Run the setup script:**
   ```bash
   python setup.py
   ```
   This will automatically install all dependencies and required NLTK resources.

4. **Set up your Google Gemini API Key:**

   Open the `gpt_predictor.py` file in a text editor and replace `"YOUR_API_KEY_HERE"` with your actual API key:
   
   ```python
   # Hardcoded API key (replace with your actual API key)
   GOOGLE_API_KEY = "your_actual_api_key_here"
   ```

## Running the Predictor

1. **Run the application:**

   **For command-line interface:**
   ```bash
   python main.py
   ```
   
   **For web application:**
   ```bash
   streamlit run app.py
   ```

2. **Follow the prompts:**
   - You will be asked to enter your dream description. Be as detailed as you like!
   - You will then be prompted to answer questions about your personality (intuition, stress level, creativity, and analytical thinking) on a scale of 1 to 10.
   - The application will analyze your dream using enhanced emotion detection and comprehensive symbol analysis.

3. **View the results:**
   - Extracted Dream Themes
   - Dream Sentiment Scores
   - Detailed Emotion Analysis with intensity levels
   - Dream Symbol Interpretations
   - Interactive Visualizations
   - Pattern Analysis of recurring themes and emotions
   - Future Influence Prediction generated by Gemini Pro
   - Personalized recommendations based on emotions and symbols

## Features and Usage

### Web Application (Streamlit)

The web application provides a more interactive experience with the following pages:

1. **Dream Input:**
   - Enter your dream description and date
   - View detailed analysis including themes, sentiment, emotions, and symbols
   - See visualizations like word clouds
   - Get AI-powered predictions and personalized recommendations

2. **Dream History:**
   - View all your recorded dreams
   - Filter dreams by date range and category
   - Visualize sentiment over time
   - See dream category distribution

3. **Analysis & Insights:**
   - View dream patterns analysis
   - Get personalized recommendations
   - Explore various visualizations on different tabs:
     - Sentiment Analysis
     - Emotion Analysis
     - Theme Analysis
     - Dashboard
   - Track theme evolution over time

4. **Settings:**
   - Update your personality profile
   - Manage your dream history data
   - Configure your API key

### Command Line Interface

The command-line interface provides a quick way to analyze a single dream without starting the web application:

```
$ python main.py

=== Future Dream Influence Predictor ===

Enter your dream description: I dreamed I was flying over a lush green forest, but suddenly a storm appeared and I started falling. I felt scared and lost.

Analyzing your dream...

=== Dream Analysis ===

Extracted Dream Themes: flying, lush, green, forest, storm, falling, scared, lost
Higher-Level Dream Themes: adventure, escape, fear

Dream Sentiment:
  Compound Score: -0.18
  Positive: 0.20
  Neutral: 0.54
  Negative: 0.26

Primary Emotions Detected: fear, sadness

Significant Dream Symbols: flying, forest, storm, falling

Symbol Interpretation:
Flying: Freedom, perspective, transcendence
Forest: Unconscious mind, unknown, mystery
...
```

## Dependencies

All required dependencies will be automatically installed by the setup script. The main dependencies include:

- google-generativeai: For interacting with Google Gemini models
- spacy: For advanced Natural Language Processing tasks
- nltk: For sentiment analysis and other NLP tasks
- pandas: For data manipulation and working with dream logs
- matplotlib & seaborn: For data visualizations
- wordcloud: For generating word cloud visualizations
- plotly: For creating interactive visualizations
- streamlit: For the web application interface

## Troubleshooting

- **Missing API Key:** If you see a message about missing API key, make sure you've replaced the placeholder in the `gpt_predictor.py` file with your actual API key.
- **SpaCy Model Error:** If you encounter issues with the spaCy model, try reinstalling it with: `python -m spacy download en_core_web_sm`
- **Visualization Issues:** Make sure you have matplotlib and other visualization libraries properly installed.
- **Dream History CSV Issues:** If you encounter errors with the dream history, check that your `dream_log.csv` file is not corrupted.

## Contributing

*Contributions are welcome! If you have ideas for improvements, bug fixes, or new features, please feel free to:*

- Fork the repository
- Create a new branch for your feature (`git checkout -b feature/your-feature`)
- Commit your changes (`git commit -am 'Add some feature'`)
- Push to the branch (`git push origin feature/your-feature`)
- Open a Pull Request

## Author
**Dhrrishit Deka**
- **Website**: https://craftedcodex.pages.dev/
- **Email**: dhrrishit@gmail.com
- **GitHub**: https://github.com/dhrrishit
- **LinkedIn**: https://www.linkedin.com/in/dhrrishitdeka/
- **X (Twitter)**: https://x.com/dhrrishitdeka
