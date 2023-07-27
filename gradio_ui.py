import gradio as gr

from utils import get_predictions

demo = gr.Interface(
    fn=get_predictions,
    inputs=gr.components.Textbox(label='Input'),
    outputs=gr.components.Label(label='Predictions', num_top_classes=5),
    allow_flagging='never'
)
