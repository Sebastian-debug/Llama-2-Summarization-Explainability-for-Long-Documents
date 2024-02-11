from pdfminer.high_level import extract_text
from text_processing import clean_summary, tokenize_sentences
from summarization import generate_summary, generate_final_summary
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re


def process_pdf(pdf_file_path, chunk_size=4096, chunk_overlap=50):
    """
    Processes a PDF file to extract text, identify key sections, split the text into chunks,
    generate summaries for each chunk, and finally combine these summaries.

    Parameters:
        pdf_file_path (str): Path to the PDF file.
        chunk_size (int): The size of each text chunk for summarization.
        chunk_overlap (int): The overlap between text chunks.

    Returns:
        tuple: A tuple containing a list of tokenized sentences from the final summary, the final summary as a string,
         and the extracted text as a string.
    """
    text = extract_text(pdf_file_path)

    # Cut off text at the References or Bibliography index if found
    pattern = re.compile(r'(\n|\r|\f)+(References|Bibliography)', flags=re.IGNORECASE)
    matches = pattern.finditer(text)
    references_index = None
    for match in matches:
        references_index = match.start()
    if references_index is not None:
        text = text[:references_index].strip()

    pattern_headings = re.compile(
        r'(\A|\n|\r|\f)+([1-9]\s+)?(ABSTRACT|INTRODUCTION|CONCLUSION|BACKGROUND|RELATED WORK)',
        flags=re.IGNORECASE)
    text = pattern_headings.sub(r'\n.\n\2\3\n.\n', text)

    # Split text into chunks for summarization
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=50, length_function=len)
    chunks = text_splitter.split_text(text)
    chunk_summaries = []
    min_len = 4
    chunk_summaries_len = len(chunks)
    if chunk_summaries_len < min_len:
        chunk_summaries_len = min_len
    for chunk in chunks:
        summary = generate_summary(chunk, chunk_summaries_len // 2)
        summary = clean_summary(summary)
        chunk_summaries.append(summary)

    # Combine chunk summaries and generate final summary
    combined_summary = "\n".join(chunk_summaries)
    final_summary = generate_final_summary(combined_summary, chunk_summaries_len)
    final_summary_cleaned = clean_summary(final_summary)

    sentences = tokenize_sentences(final_summary_cleaned)

    return sentences, final_summary_cleaned, text


