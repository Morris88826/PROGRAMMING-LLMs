from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Load the LLaMA model and tokenizer
model_name = "decapoda-research/llama-7b-hf"  # Replace with your LLaMA model name/path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

@app.route('/v1/completions', methods=['POST'])
def generate_completion():
    data = request.json
    prompt = data.get("prompt")
    stop = data.get("stop", [])

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate the output
    outputs = model.generate(**inputs, max_new_tokens=100)

    # Decode the output to text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove text after any stop sequences
    for stop_sequence in stop:
        generated_text = generated_text.split(stop_sequence)[0]

    # Return the response in the format expected
    return jsonify({"choices": [{"text": generated_text}]})

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=11434)
