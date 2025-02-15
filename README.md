# Future Influence Predictor

![Future Influence Predictor Banner](banner.png)
> **Note:** The banner above is a text-based image (ASCII art or a text graphic) used as a visual placeholder. It does not contain any graphical content beyond stylized text.

## Project Description

**Future Influence Predictor** is an experimental Python project that analyzes your dream descriptions along with personality data to predict how recurring dream elements might influence your future decisions. The application leverages Natural Language Processing (NLP) with spaCy and NLTK (VADER) to extract themes and sentiments from your dreams, and then uses a GPT-based model (via OpenAI) to generate creative, personalized predictions.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Running the CLI Application](#running-the-cli-application)
  - [Optional: Running the Streamlit Interface](#optional-running-the-streamlit-interface)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features

- **Dream Analysis:** Extracts recurring themes from dream descriptions using spaCy.
- **Sentiment Evaluation:** Analyzes dream mood with NLTK's VADER sentiment analyzer.
- **Personality Insights:** Collects basic personality data via a simple questionnaire.
- **Future Predictions:** Generates GPT-based predictions correlating dream themes with personality data.
- **Visualization:** Displays dream sentiment trends over time and creates word clouds of extracted themes.
- **Data Logging:** Optionally logs dream entries to a CSV file for tracking recurring patterns.

## Requirements

- **Python Version:** 3.7 or higher
- **Key Libraries:**  
  - `openai` (for GPT-based predictions)
  - `spacy` (for NLP processing)
  - `nltk` (for sentiment analysis; ensure to download the `vader_lexicon` resource)
  - `pandas` (for data handling)
  - `matplotlib` & `seaborn` (for visualization)
  - `wordcloud` (for generating word clouds)
  - `streamlit` (optional, for a web-based interface)

## Installation

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/future-influence-predictor.git
    cd future-influence-predictor
    ```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download spaCy's English Model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

4. **Download NLTK VADER Lexicon:**
    Run the following commands in your Python interpreter or include them in your script:
    ```python
    import nltk
    nltk.download('vader_lexicon')
    ```

## Project Structure

