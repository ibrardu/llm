import os
from flask import Flask, request, jsonify, render_template, redirect, url_for
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from werkzeug.utils import secure_filename
import threading
import train_model

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload and model directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Global variable to store available models
def update_available_models():
    return [name for name in os.listdir(MODEL_FOLDER) if os.path.isdir(os.path.join(MODEL_FOLDER, name))]

available_models = update_available_models()

@app.route('/')
def index():
    global available_models
    available_models = update_available_models()
    return render_template('index.html', models=available_models)

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            model_name = request.form['model_name']

            # Start a new thread for training
            thread = threading.Thread(target=train_model.train, args=(file_path, model_name))
            thread.start()

            return redirect(url_for('index'))
    return render_template('train.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    model_name = request.form['model']
    model_path = os.path.join(MODEL_FOLDER, model_name)

    # Load the selected model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
