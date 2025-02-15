# Future Dream Influence Predictor

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) <!-- Optional: Add a LICENSE file and link here -->

## Overview

The **Future Dream Influence Predictor** is a Python-based application that analyzes your dreams to extract key themes, assess sentiment, and creatively predict how these dream elements might influence your future decisions.  Leveraging Natural Language Processing (NLP) with spaCy and NLTK, sentiment analysis with VADER, and the powerful Gemini Pro model from Google AI, this project offers a fun and insightful look into the potential impact of your subconscious.

**Imagine:** You've just had a vivid dream. You describe it to the predictor, and it not only tells you the dominant themes and the overall feeling of the dream but also gives you a creative glimpse into how these dream elements might play out in your waking life.

**This project features:**

* **Dream Theme Extraction:** Uses spaCy to identify and extract significant keywords (nouns, verbs, adjectives) from your dream description.
* **Sentiment Analysis:** Employs NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) to analyze the sentiment expressed in your dream text, providing scores for positive, negative, neutral, and compound sentiment.
* **Future Influence Prediction:** Integrates with Google's Gemini Pro model to generate creative and thoughtful predictions about how your dream themes and sentiment could influence your future decisions, taking into account your personality traits (intuition and stress level).
* **Interactive Personality Input:** Gathers basic personality data (intuition and stress levels) through a simple interactive questionnaire to personalize the future influence prediction.
* **Visualizations:**
    * **Word Cloud:** Generates a visually appealing word cloud from the extracted dream themes, highlighting the most prominent keywords.
    * **Sentiment Over Time Plot:** (If you log multiple dreams) Creates a line plot showing the trend of dream sentiment over time, allowing you to visualize emotional patterns in your dreams.
* **Dream Logging:** Saves your dream descriptions, extracted themes, and sentiment scores to a CSV file (`dream_log.csv`), enabling you to track your dreams and analyze trends over time.

## Key Features

* **NLP Powered Analysis:** Utilizes spaCy and NLTK for robust dream text processing.
* **Gemini Pro Integration:** Leverages Google's advanced language model for creative future predictions.
* **Personalized Predictions:** Considers user's personality traits for more tailored insights.
* **Visual Dream Summaries:**  Provides word clouds and sentiment plots for visual understanding of dream data.
* **Simple and Interactive:** Easy-to-use command-line interface for dream input and results.

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

   ```
GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
```
