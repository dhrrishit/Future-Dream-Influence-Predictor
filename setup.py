import subprocess
import sys
import importlib.util
import os

def check_package_installed(package_name):
    return importlib.util.find_spec(package_name) is not None

def install_package(package_name):
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Successfully installed {package_name}!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package_name}: {e}")
        return False

def download_nltk_resources():
    import nltk
    print("Downloading required NLTK resources...")
    nltk_resources = ['punkt', 'punkt_tab', 'vader_lexicon', 'stopwords', 'wordnet']
    for resource in nltk_resources:
        print(f"Downloading NLTK resource: {resource}")
        try:
            nltk.download(resource, quiet=False)
            if resource == 'punkt':
                try:
                    nltk.data.find('tokenizers/punkt')
                    nltk.data.find('tokenizers/punkt/english.pickle')
                    print(f"Successfully verified NLTK resource: {resource}")
                except LookupError:
                    print(f"Resource {resource} was downloaded but verification failed. Trying again...")
                    nltk.download(resource, download_dir=nltk.data.path[0], force=True)
            else:
                print(f"Successfully downloaded NLTK resource: {resource}")
        except Exception as e:
            print(f"Error downloading NLTK resource {resource}: {e}")
            return False
    return True

def main():
    if not check_package_installed("nltk"):
        if not install_package("nltk"):
            print("Failed to install NLTK. Please install it manually using 'pip install nltk'.")
            sys.exit(1)
    
    if not download_nltk_resources():
        print("Failed to download some NLTK resources. Please try downloading them manually.")
        print("You can do this by running Python and executing:")
        print("import nltk; nltk.download('punkt'); nltk.download('vader_lexicon'); nltk.download('stopwords'); nltk.download('wordnet')")
    
    if not check_package_installed("spacy"):
        if not install_package("spacy"):
            print("Failed to install spaCy. Please install it manually using 'pip install spacy'.")
            sys.exit(1)
    
    print("Installing spaCy model 'en_core_web_sm'...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("Successfully installed 'en_core_web_sm' model!")
        
        try:
            subprocess.check_call([sys.executable, "-c", "import spacy; nlp = spacy.load('en_core_web_sm')"])
            print("Verified 'en_core_web_sm' model is working correctly!")
        except subprocess.CalledProcessError:
            print("Model may have installed but failed verification. Trying alternative installation method...")
            install_package("https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl")
    except subprocess.CalledProcessError as e:
        print(f"Error installing spaCy model using standard method: {e}")
        print("Trying alternative installation method...")
        if not install_package("https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0-py3-none-any.whl"):
            print("Failed to install the spaCy model. Please try installing it manually.")
            sys.exit(1)

if __name__ == "__main__":
    main()