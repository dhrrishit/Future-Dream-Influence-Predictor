Future Influence Predictor

Note: The banner above is a text-based image (e.g., ASCII art or a text graphic) used as a visual placeholder. It contains only stylized text.

Project Description

Future Influence Predictor is an experimental Python project that analyzes your dream descriptions along with personality data to predict how recurring dream elements might influence your future decisions. The application leverages Natural Language Processing (NLP) with spaCy and NLTK (VADER) to extract themes and sentiments from your dreams and then uses a GPT-based model (via OpenAI) to generate creative, personalized predictions.

Table of Contents

- Features
- Requirements
- Installation
- Project Structure
- Usage
  - Running the CLI Application
  - Optional: Running the Streamlit Interface
- Configuration
- Contributing
- License
- Contact

Features

- Dream Analysis: Extracts recurring themes from dream descriptions using spaCy.
- Sentiment Evaluation: Analyzes dream mood with NLTK's VADER sentiment analyzer.
- Personality Insights: Collects basic personality data via a simple questionnaire.
- Future Predictions: Generates GPT-based predictions correlating dream themes with personality data, powered by a GPT model (via OpenAI).
- Visualization: Displays dream sentiment trends over time and creates word clouds of extracted themes to aid in understanding dream patterns.
- Data Logging: Optionally logs dream entries to a CSV file (dream_log.csv) for tracking recurring patterns and building a personal dream journal.

Requirements

- Python Version: 3.7 or higher
- Key Libraries:
  - openai (for GPT-based predictions)
  - spacy (for NLP processing)
  - nltk (for sentiment analysis; ensure you download the vader_lexicon resource)
  - pandas (for data handling and CSV operations)
  - matplotlib & seaborn (for data visualization)
  - wordcloud (for generating word clouds)
  - streamlit (optional, for a web-based interactive interface)

Installation

1. Clone the Repository:
    git clone https://github.com/yourusername/future-influence-predictor.git
    cd future-influence-predictor
    Replace yourusername with your actual GitHub username or the repository owner's username.

2. Install Dependencies:
    pip install -r requirements.txt
    This command will install all the necessary Python libraries listed in the requirements.txt file.

3. Download spaCy's English Model:
    python -m spacy download en_core_web_sm
    This downloads the small English language model for spaCy, which is used for NLP tasks.

4. Download NLTK VADER Lexicon:
    Run the following commands in your Python interpreter or include them in your script once:
    import nltk
    nltk.download('vader_lexicon')
    This downloads the VADER lexicon, required for sentiment analysis using NLTK.

Project Structure

future_influence_predictor/
├── requirements.txt       # List of project dependencies
├── main.py                # Main application script (CLI interface)
├── nlp_utils.py           # NLP utility functions (theme extraction, sentiment analysis)
├── gpt_predictor.py       # Module for GPT-based future influence prediction
├── visualization.py       # Visualization utilities (sentiment trends, word clouds)
├── personality.py         # Module for collecting personality data via questionnaire
├── app.py                 # (Optional) Streamlit web interface for interactive usage
└── dream_log.csv          # (Optional) CSV file for logging dream entries

Usage

Running the CLI Application

To run the application from the command line, navigate to the project directory and execute:

python main.py

The application will then guide you through the following steps:

1. Enter your dream description: You'll be prompted to input a detailed description of your dream.
2. Answer personality questions:  You'll be asked a few personality-related questions to provide context for the prediction.

After providing the input, the application will:

- Extract key themes and analyze the sentiment of your dream description.
- Generate a personalized future influence prediction based on your dream themes, sentiment, and personality data using a GPT model.
- Optionally log your dream entry to dream_log.csv if you choose to enable data logging.
- If multiple dream entries exist in dream_log.csv, it will generate and display visualizations of sentiment trends and word clouds.

Optional: Running the Streamlit Interface

For a more interactive, web-based experience, you can run the Streamlit interface. Ensure you have Streamlit installed (pip install streamlit) and then execute:

streamlit run app.py

This command will launch the Streamlit application in your default web browser. You can then input your dream details, provide personality scores through an interactive interface, and view the generated prediction and visualizations directly in the browser.

Configuration

OpenAI API Key:

To use the GPT-based prediction feature, you need to configure your OpenAI API key.

1. Locate gpt_predictor.py: Open the gpt_predictor.py file in your project directory.
2. Set your API key: Find the line where the OpenAI API key is set (likely within the generate_prediction function or similar). Replace "YOUR_OPENAI_API_KEY" with your actual OpenAI API key from OpenAI's website (https://platform.openai.com/account/api-keys).

    openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your actual API key

OpenAI API Migration Notice:

Important:  The OpenAI Python library has undergone significant updates. If you encounter issues related to API compatibility, especially if you are using openai version 1.0.0 or higher, please consider the following:

Option 1: Automatic Migration (Recommended for newer OpenAI versions):

Run the OpenAI migration tool in your terminal:

openai migrate

This command attempts to automatically update your code to be compatible with the latest OpenAI API interface.

Option 2: Pin to an Older OpenAI Version (If migration is not desired):

If you prefer to use the older API interface (compatible with openai version 0.28), you can pin your OpenAI package version:

pip install openai==0.28
This will downgrade or install version 0.28 of the openai library. Choose the option that best suits your needs and OpenAI library version.

Personality Data:

You can customize the personality questionnaire used in the application.

- Edit personality.py: Open the personality.py file to modify the questions asked to the user. You can adjust the questions to focus on specific personality traits or aspects you want to correlate with dream influence.

Visualization:

Visualization settings, such as plot styles and word cloud appearance, can be adjusted in visualization.py.

- Modify visualization.py:  Customize the functions in visualization.py to change colors, fonts, word cloud parameters, and other visual elements to match your preferences.

Contributing

Contributions are welcome and encouraged! If you have ideas for improvements, bug fixes, or new features, please follow these steps:

1. Fork the repository: Click the "Fork" button on the GitHub repository page to create your own copy of the project.
2. Create your feature branch:
    git checkout -b feature/my-new-feature
    Replace my-new-feature with a descriptive name for your branch.
3. Commit your changes:
    git commit -am 'Add a descriptive commit message about your changes'
4. Push to the branch:
    git push origin feature/my-new-feature
5. Create a new Pull Request: Go to your forked repository on GitHub and click the "Create Pull Request" button. Provide a clear title and description of your changes in the pull request.



## Contact

For questions, feedback, or suggestions regarding this project, please contact:

Dhrrishit Deka: dhrrishitdeka13@gmail.com
GitHub Profile: https://github.com/dhrrishit 

