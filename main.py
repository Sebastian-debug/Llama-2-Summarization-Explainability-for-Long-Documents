import gradio as gr
from pdf_processing import process_pdf
from similarity_analysis import find_most_similar_sentence


def process_and_display(file_obj):
    """
      Processes the uploaded PDF file and updates the interface with the summary sentences, full summary, and full text.

      Parameters:
        file_obj (str): The path to the PDF file uploaded by the user.

      Returns:
        tuple: A tuple containing the update function for the Gradio interface, the summary, and the full text of the PDF.
    """
    try:
        summary_sentences, summary, text = process_pdf(file_obj)
        return gr.update(choices=summary_sentences), summary, text
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return gr.update(choices=[]), "Error in processing PDF", ""


def on_sentence_select(sentence, full_text):
    """
      Finds and returns the most similar sentence from the text based on the selected summary sentence.

      Parameters:
        sentence (str): A sentence selected from the summary.
        full_text (str): The text of the document.

      Returns:
        str: The most similar sentence found in the full text.
    """
    try:
        most_similar = find_most_similar_sentence(sentence, full_text)
        return most_similar
    except Exception as e:
        print(f"Error finding similar sentence: {e}")
        return "Error in finding similar sentence"

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload a PDF file", type="filepath", file_types=[".pdf"])
            submit_btn = gr.Button("Submit")
            similar_sentence_output = gr.Textbox(label="Most similar sentence in the original document",
                                                 interactive=False)
            summary_heading = gr.HTML("<h2>Summary</h2>")
            full_summary_output = gr.HTML()
        with gr.Column():
            summary_output = gr.Radio(label="Select a sentence from the summary", choices=[], interactive=True)
            full_text_output = gr.Textbox(label="Full Text Output", interactive=False, visible=False)

    submit_btn.click(
        process_and_display,
        inputs=[file_input],
        outputs=[summary_output, full_summary_output, full_text_output]
    )


    summary_output.change(on_sentence_select, inputs=[summary_output, full_text_output], outputs=[similar_sentence_output])

demo.launch(debug=True, share=True)
