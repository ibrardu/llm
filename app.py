from flask import Flask , request , render_template , redirect , url_for
from transformers import pipeline , GPT2LMHeadModel , GPT2Tokenizer
import os
import torch

app = Flask ( __name__ )
app.config['MODEL_FOLDER'] = 'models/'
os.makedirs ( app.config['MODEL_FOLDER'] , exist_ok=True )

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained ( 'gpt2' )


@app.route ( '/' )
def index() :
    # Get the list of trained models
    models = [f for f in os.listdir ( app.config['MODEL_FOLDER'] ) if f.endswith ( '.pt' )]
    return render_template ( 'index.html' , models=models )


@app.route ( '/generate' , methods=['POST'] )
def generate() :
    prompt = request.form['prompt']
    model_name = request.form['model']

    # Load the selected model
    model = GPT2LMHeadModel.from_pretrained ( 'gpt2' )
    model_path = os.path.join ( app.config['MODEL_FOLDER'] , model_name )
    model.load_state_dict ( torch.load ( model_path ) )
    model.eval ( )

    # Generate text using the selected model
    generator = pipeline ( 'text-generation' , model=model , tokenizer=tokenizer )
    result = generator ( prompt , max_length=100 , num_return_sequences=1 )
    return render_template ( 'result.html' , result=result[0]['generated_text'] )


if __name__ == '__main__' :
    app.run ( debug=True )
