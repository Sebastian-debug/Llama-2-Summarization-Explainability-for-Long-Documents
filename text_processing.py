import re
import nltk
import nltk.data
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

def tokenize_sentences(text):
    """
    Tokenizes a text into sentences considering special abbreviations.

    Parameters:
        text (str): The text to tokenize.

    Returns:
        list: A list of tokenized sentences.
    """
    punkt_param = PunktParameters()
    abbreviation = ['al', 'et al', 'etc', 'e.g', 'i.e', 'dr']
    punkt_param.abbrev_types = set(abbreviation)
    tokenizer_sentence = PunktSentenceTokenizer(punkt_param)
    sentences = tokenizer_sentence.tokenize(text)
    return sentences

def remove_closing_statements(text):
    """
    Removes standard closing statements from a text.

    Parameters:
        - text (str): The text to process.

    Returns:
        - str: The text with closing statements removed.
    """
    sentences = tokenize_sentences(text)

    unwanted_closings = [
        "I hope this helps!",
        "Let me know if you have any questions or need further clarification."
    ]

    while sentences and sentences[-1] in unwanted_closings:
        sentences.pop()

    return ' '.join(sentences)

def clean_summary(summary):
    """
        Cleans a summary text by removing list indicators and unwanted characters.

    Parameters:
        summary (str): The summary text to clean.

    Returns:
        str: The cleaned summary.
    """

    cleaned_summary = re.sub(r'\s*\d+\.\s*', ' ', summary)
    cleaned_summary = re.sub(r'\s*\d+\)\s*', ' ', cleaned_summary)
    cleaned_summary = re.sub(r'\*+\s*', ' ', cleaned_summary)

    cleaned_summary = re.sub(r'\s+', ' ', cleaned_summary).strip()
    cleaned_summary = remove_closing_statements(cleaned_summary)

    return cleaned_summary

def clean_source_text(text):
    """
        Prepares source text by removing hyphenations at line breaks and merging multiple newlines.

    Parameters:
        text (str): The source text to clean.

    Returns:
        str: The cleaned source text.
    """
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n+', ' ', text)
    return text

