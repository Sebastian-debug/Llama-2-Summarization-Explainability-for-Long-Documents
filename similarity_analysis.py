import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from text_processing import clean_source_text, tokenize_sentences

# Load the sentence transformer model
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model.to(device)
except Exception as e:
    raise RuntimeError(f"Failed to load model or tokenizer: {e}")


def mean_pooling(model_output, mask):
    """
    Applies mean pooling to the model's output to get sentence embeddings.

    Parameters:
        model_output (Tensor): The output from the transformer model.
        mask (Tensor): The attention mask to apply over the model output.

    Returns:
        Tensor: The mean-pooled sentence embeddings.
    """
    token_embeddings = model_output[0]
    mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)

def encode_and_embed(sentences):
    """
    Encodes and computes embeddings for given sentences.

    Parameters:
        sentences (str or list of str): The sentence(s) to encode and embed.

    Returns:
        Tensor: The embeddings of the input sentence(s).
    """
    encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded)
    return mean_pooling(model_output, encoded['attention_mask'])

def find_most_similar_sentence(clicked_sentence, full_text):
    """
    Identifies the most similar sentence in a given text to the specified 'clicked' sentence.

    Parameters:
        clicked_sentence (str): The sentence to compare against.
        full_text (str): The text to search for a similar sentence

    Returns:
        str: The most similar sentence found in the text.
    """

    if not full_text.strip():
        raise ValueError("The full text is empty or only contains whitespace.")

    preprocessed_text = clean_source_text(full_text)
    source_sentences = tokenize_sentences(preprocessed_text)

    if not source_sentences:
        raise ValueError("No sentences found after tokenization.")

    encoded_clicked = tokenizer(clicked_sentence, padding=True, truncation=True, return_tensors='pt').to(device)
    encoded_sources = tokenizer(source_sentences, padding=True, truncation=True, return_tensors='pt').to(device)

    clicked_embedding = encode_and_embed(clicked_sentence)
    source_embeddings = encode_and_embed(source_sentences)

    # Calculate cosine similarities
    clicked_embedding = F.normalize(clicked_embedding, p=2, dim=1)
    source_embeddings = F.normalize(source_embeddings, p=2, dim=1)
    cos_scores = torch.mm(clicked_embedding, source_embeddings.T).squeeze(0)
    most_similar_idx = torch.argmax(cos_scores).item()

    return source_sentences[most_similar_idx]


