import os
import sys
import torch
from transformers import GPT2LMHeadModel , GPT2Tokenizer

# Set the cache directory to a subdirectory in your project directory before any imports from huggingface
cache_dir = os.path.join ( os.getcwd ( ) , 'huggingface_cache' )
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
os.makedirs ( cache_dir , exist_ok=True )


def load_model(model_path) :
    tokenizer = GPT2Tokenizer.from_pretrained ( 'gpt2' , cache_dir=cache_dir )
    model = GPT2LMHeadModel.from_pretrained ( 'gpt2' , cache_dir=cache_dir )
    model.load_state_dict ( torch.load ( model_path ) )
    model.eval ( )
    return tokenizer , model


def generate_text(tokenizer , model , prompt , max_length=100) :
    inputs = tokenizer.encode ( prompt , return_tensors='pt' )
    outputs = model.generate ( inputs , max_length=max_length )
    return tokenizer.decode ( outputs[0] , skip_special_tokens=True )


def select_model(models_dir) :
    models = [f for f in os.listdir ( models_dir ) if f.endswith ( '.pt' )]
    if not models :
        print ( "No models found in the specified directory." )
        sys.exit ( 1 )

    for idx , model in enumerate ( models ) :
        print ( f"{idx + 1}: {model}" )

    choice = int ( input ( "Select a model by number: " ) ) - 1
    if choice < 0 or choice >= len ( models ) :
        print ( "Invalid model number selected." )
        sys.exit ( 1 )

    return os.path.join ( models_dir , models[choice] )


if __name__ == "__main__" :
    models_dir = 'models'
    model_path = select_model ( models_dir )
    prompt = input ( "Enter your prompt: " )

    tokenizer , model = load_model ( model_path )
    generated_text = generate_text ( tokenizer , model , prompt )
    print ( "Generated Text:" )
    print ( generated_text )
