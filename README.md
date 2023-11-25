# Text Paraphrasing Tool using Transformers

This is a Python script that uses the GPT-3.5 model to paraphrase input text. It leverages the Hugging Face `transformers` library for easy integration with the GPT-3.5 model.

## Requirements

- Python 
- Install the required dependencies by running:

  ```bash
  pip install -r requirements.txt
  ```

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/NoorMahammad-S/text-paraphrasing-tool.git
   ```

2. Change into the project directory:

   ```bash
   cd text-paraphrasing-tool
   ```

3. Run the paraphrasing tool:

   ```bash
   python paraphrase_tool.py
   ```

   Follow the on-screen instructions to enter text for paraphrasing.

## Configuration

- The GPT-3.5 model is loaded using the Hugging Face `transformers` library. You can change the model and tokenizer by updating the `model_name` variable in the 'main.py` script.

## Additional Features

- Exception handling for user input and paraphrasing errors.
- Handling empty input to ensure a valid text entry.

## Acknowledgments

- The GPT-3.5 model is provided by EleutherAI and can be found on the Hugging Face Model Hub.

Feel free to fork, modify, and use this code according to your needs. If you encounter any issues or have suggestions for improvement, please open an issue or submit a pull request.

Happy paraphrasing!
