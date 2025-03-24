import subprocess
import sys
import importlib.util
import os
import pkg_resources

# App version, keep in sync with app.py
APP_VERSION = "1.0.0"

def check_package_installed(package_name):
    """Check if a package is installed and return its version if it is."""
    try:
        package = pkg_resources.get_distribution(package_name)
        return package.version
    except pkg_resources.DistributionNotFound:
        return None

def install_package(package_name, version=None):
    """Install a Python package with optional version specification."""
    print(f"Installing {package_name}...")
    package_spec = package_name
    if version:
        package_spec = f"{package_name}=={version}"
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
        print(f"Successfully installed {package_name}!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package_name}: {e}")
        return False

def download_nltk_resources():
    """Download all required NLTK resources."""
    try:
        import nltk
        print("Downloading required NLTK resources...")
        nltk_resources = ['punkt', 'vader_lexicon', 'stopwords', 'wordnet']
        
        for resource in nltk_resources:
            try:
                # First check if resource already exists
                nltk.data.find(f"{resource}")
                print(f"NLTK resource already available: {resource}")
            except LookupError:
                print(f"Downloading NLTK resource: {resource}")
                try:
                    nltk.download(resource, quiet=True)
                    print(f"Successfully downloaded NLTK resource: {resource}")
                except Exception as e:
                    print(f"Error downloading NLTK resource {resource}: {e}")
                    print(f"Will attempt to continue without {resource}.")
        
        return True
    except Exception as e:
        print(f"Error setting up NLTK resources: {e}")
        return False

def install_spacy_model():
    """Install the spaCy English language model."""
    print("Installing spaCy model 'en_core_web_sm'...")
    try:
        # Check if model is already installed
        try:
            subprocess.check_call([sys.executable, "-c", "import spacy; nlp = spacy.load('en_core_web_sm')"], 
                                  stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            print("SpaCy model 'en_core_web_sm' already installed!")
            return True
        except (subprocess.CalledProcessError, ImportError):
            pass  # Not installed, continue with installation
        
        try:
            # First try the recommended way
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            print("Successfully installed 'en_core_web_sm' model!")
            return True
        except subprocess.CalledProcessError:
            # If that failed, try direct download
            print("Standard installation failed. Trying direct download...")
            model_url = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
            if install_package(model_url):
                return True
            
            # If that also failed, try an older version
            print("Trying older model version...")
            model_url = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl"
            return install_package(model_url)
    except Exception as e:
        print(f"Error installing spaCy model: {e}")
        return False

def install_dependencies():
    """Install all required dependencies from requirements.txt."""
    print("Installing dependencies from requirements.txt...")
    requirements_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    
    if not os.path.exists(requirements_file):
        print("Error: requirements.txt not found!")
        return False
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("Successfully installed dependencies from requirements.txt!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies from requirements.txt: {e}")
        
        # Try installing one by one
        print("Trying to install packages individually...")
        success = True
        
        # Core dependencies with minimum versions
        dependencies = [
            ("streamlit", "1.18.0"), 
            ("nltk", "3.8.1"), 
            ("pandas", "1.5.3"),
            ("matplotlib", "3.7.2"),
            ("seaborn", "0.12.2"),
            ("wordcloud", "1.8.2.2"),
            ("plotly", "5.14.1"),
            ("spacy", "3.7.2"),
            ("google-generativeai", "0.3.1")
        ]
        
        for package, version in dependencies:
            if not check_package_installed(package):
                if not install_package(package, version):
                    print(f"Failed to install {package}")
                    success = False
        
        return success

def create_empty_dream_log():
    """Create an empty dream log CSV file if it doesn't exist."""
    if not os.path.exists("dream_log.csv"):
        print("Creating empty dream log file...")
        try:
            import pandas as pd
            empty_df = pd.DataFrame(columns=["date", "dream", "themes", "sentiment", "category", "emotions", "symbols"])
            empty_df.to_csv("dream_log.csv", index=False)
            print("Created empty dream log file.")
        except Exception as e:
            print(f"Error creating dream log file: {e}")

def update_api_key_reminder():
    """Remind the user to set their API key in the gpt_predictor.py file."""
    print("\nIMPORTANT: To use the AI prediction features, you need to:")
    print("1. Get a Google Gemini API key from https://makersuite.google.com/app/apikey")
    print("2. Open the gpt_predictor.py file")
    print("3. Replace 'YOUR_API_KEY_HERE' with your actual API key")
    print("Example: GOOGLE_API_KEY = \"abc123your-actual-api-key-here\"")

def main():
    """Main setup function."""
    print(f"Setting up Future Dream Influence Predictor v{APP_VERSION}...")
    
    # Install all dependencies
    if not install_dependencies():
        print("Warning: Some dependencies may not have been installed correctly.")
        print("You may need to install them manually.")
    
    # Download NLTK resources
    if not download_nltk_resources():
        print("Warning: NLTK resource download had issues.")
    
    # Install spaCy model
    if not install_spacy_model():
        print("Warning: Could not install the spaCy model.")
        print("You will need to install it manually with:")
        print("python -m spacy download en_core_web_sm")
    
    # Create empty dream log if needed
    create_empty_dream_log()
    
    # Remind about API key setup
    update_api_key_reminder()
    
    print("\nSetup completed!")
    print("To run the application, use: streamlit run app.py")
    print("Or for command-line version: python main.py")

if __name__ == "__main__":
    main()