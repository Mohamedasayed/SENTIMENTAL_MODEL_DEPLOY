import gradio as gr
from transformers import pipeline


classifier = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")


def predict_sentiment(text):
    result = classifier(text)[0]
    return result


app = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(),
    outputs="json"
)


app.launch()
