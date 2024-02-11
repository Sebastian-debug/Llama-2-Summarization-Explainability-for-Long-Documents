# Llama 2 Summarization & Explainability for Long Documents

## Overview
Llama 2-7B-Chat is used for advanced summaries enhanced by the SBERT-based all-MiniLM-L6-v2 model to identify key insights from long scientific texts.

This project is designed to process, summarize, and analyze texts from PDF documents. It generates summaries, and finds similar sentences within the text.

## Video

You can watch a tutorial video here: https://www.youtube.com/watch?v=cS6Kj9vsGJg 

![BERT_Up1](https://github.com/Sebastian-debug/Llama-2-Summarization-Explainability-for-Long-Documents/assets/56771437/63b14970-52ea-4ce7-9bd2-bee9eb220148)

## Installation

Set up "config.json" with your Huggingface API token before running the application.

Ensure Python is installed on your system and follow these steps:

git clone https://github.com/your-repository/project-name.git

Navigate to Project Directory and install the Dependencies:

`pip install -r requirements.txt`

## Usage

Meta’s large language model Llama2-7B-Chat requires around 30 GB of GPU memory. In order to fulfill the performance requirements, Google Colab Pro+
was used , which provided access to an NVIDIA A100 Tensor Core GPU with 40 GB of GPU memory.

Execute main.py to start the application:

`python main.py`

