from transformers import GPT2LMHeadModel, GPT2Tokenizer

def paraphrase_text(text, model, tokenizer, max_length=50, temperature=0.7, top_k=50):
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # Generate paraphrased text
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1,
    )

    paraphrased_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return paraphrased_text

def main():
    # Load pre-trained GPT-3.5 model and tokenizer
    model_name = "EleutherAI/gpt-neo-2.7B"  # Change this to the appropriate GPT-3.5 model
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    while True:
        # Get input text from the user
        original_text = input("Enter text to paraphrase (type 'exit' to quit): ")

        if original_text.lower() == 'exit':
            break

        # Paraphrase the input text
        paraphrased_text = paraphrase_text(original_text, model, tokenizer)

        # Print the paraphrased text
        print("\nParaphrased Text:")
        print(paraphrased_text)
        print("\n")

if __name__ == "__main__":
    main()
