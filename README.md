# LLM

This project is a Flask web application that allows users to upload books in `.txt` format for training a GPT-2 model and then use the trained models to generate text based on user prompts.

## Project Directory Structure


## Files and Directories

- `app.py`: Main Flask application file that handles routing and interactions with the trained models.
- `preprocess.py`: Script to preprocess the uploaded text files.
- `train_model.py`: Script to train a GPT-2 model on the uploaded text files.
- `static/`: Directory for static files like CSS.
  - `styles.css`: CSS file for styling the HTML pages.
- `templates/`: Directory for HTML templates.
  - `index.html`: Home page template where users can enter prompts and select a model.
  - `train.html`: Page template where users can upload text files for training new models.
  - `result.html`: Result page template (if needed).
- `uploads/`: Directory where uploaded text files are stored.
- `models/`: Directory where trained models are stored.
- `requirements.txt`: List of Python packages required for the project.
- `Procfile`: File for specifying the commands that are run by the Heroku app.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ibrardu/llm.git
   cd llm
2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
5. **Run the Flask application:**
   ```bash
   python app.py
5. **Open your web browser and navigate to `http://127.0.0.1:5000/`.**



