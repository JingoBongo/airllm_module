import gradio as gr
from airllm import HuggingFaceModelLoader, AutoModelForCausalLM

# model_loader = HuggingFaceModelLoader("meta-llama/Meta-Llama-3-70B-Instruct")
model_loader = HuggingFaceModelLoader("meta-llama/Llama-3.2-3B")
model = AutoModelForCausalLM.from_pretrained(model_loader)

def generate_text(input_text):
    input_ids = model.tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=100)
    return model.tokenizer.decode(output[0])

iface = gr.Interface(
    fn=generate_text, 
    inputs=gr.Textbox(placeholder="Enter prompt..."),
    outputs="text",
    title="LLaMA 3 70B Text Generation"
)

# example from https://gist.github.com/ruvnet/f4ac76cb411c8da0b954f91197ca1774

iface.launch(server_name="0.0.0.0", server_port=5556)