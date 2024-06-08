import os

# Set the cache directory to a subdirectory in your project directory
os.environ['TRANSFORMERS_CACHE'] = os.path.join ( os.getcwd ( ) , 'huggingface_cache' )

import torch
from transformers import GPT2LMHeadModel , GPT2Tokenizer


# Define the training function
def train_model(file_path , model_name) :
    tokenizer = GPT2Tokenizer.from_pretrained ( 'gpt2' )

    with open ( file_path , 'r' ) as f :
        text = f.read ( )

    # Tokenize and encode the text
    inputs = tokenizer ( text , return_tensors='pt' , max_length=512 , truncation=True )
    model = GPT2LMHeadModel.from_pretrained ( 'gpt2' )

    # Fine-tune the model
    model.train ( )
    optimizer = torch.optim.AdamW ( model.parameters ( ) , lr=5e-5 )

    for epoch in range ( 3 ) :  # number of epochs
        optimizer.zero_grad ( )
        outputs = model ( **inputs , labels=inputs['input_ids'] )
        loss = outputs.loss
        loss.backward ( )
        optimizer.step ( )
        print ( f"Epoch {epoch + 1}, Loss: {loss.item ( )}" )

    # Save the trained model
    model_save_path = os.path.join ( 'models' , f"{model_name}.pt" )
    os.makedirs ( os.path.dirname ( model_save_path ) , exist_ok=True )
    torch.save ( model.state_dict ( ) , model_save_path )
    print ( f"Model saved to {model_save_path}" )


# Example usage
if __name__ == '__main__' :
    file_path = 'uploads/HP1.txt'  # Update this path to your training file
    model_name = 'harrypotter'  # Update this to your desired model name
    train_model ( file_path , model_name )
