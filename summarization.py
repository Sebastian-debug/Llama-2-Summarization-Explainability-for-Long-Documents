import torch
import json
from transformers import AutoTokenizer, pipeline
from langchain import LLMChain, HuggingFacePipeline, PromptTemplate


def load_config(config_path='config.json'):
    """
    Loads the configuration from a JSON file.

    Parameters:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Config file.
    """
    try:
        with open(config_path) as config_file:
            return json.load(config_file)
    except FileNotFoundError:
        print("Config file not found.")
        return None
    except json.JSONDecodeError:
        print("Error decoding config file.")
        return None


config = load_config()
if config and 'huggingface_token' in config:
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=config['huggingface_token'])
    text_gen_pipeline = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        token=config['huggingface_token'],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=4000,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    llm = HuggingFacePipeline(pipeline=text_gen_pipeline, model_kwargs={'temperature': 0})
else:
    raise ValueError("Huggingface token missing from configuration.")


# Summarization functions
def generate_summary(text_chunk, sentence_number):
    """
    Generates a summary for a given text chunk.

    Parameters:
        text_chunk (str): The text chunk to summarize.
        sentence_number (int): The desired number of sentences in the summary.

    Returns:
        str: The generated summary.
    """

    template =f"""
          Write a concise summary of the following text delimited by triple backquotes.
          The summary should be no longer than {sentence_number} sentences and should concisely capture the main points.
          ```{{text}}```
          SUMMARY:
          """
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    summary = llm_chain.run(text_chunk)
    return summary

def generate_final_summary(text, sentence_number):
    """
    Generates the final summary of the pdf file.

    Parameters:
        text (str): The text to summarize.
        sentence_number (int): The base number of sentences for the summary.

    Returns:
        str: The final generated summary.
    """
    template =f"""
              Write a concise summary of the following text delimited by triple backquotes.
              Respond in at least {sentence_number} sentences but the summary should be no longer than {sentence_number + sentence_number//2} sentences.
              ```{{text}}```
              SUMMARY:
              """
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    summary = llm_chain.run(text)
    return summary




