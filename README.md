import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def initialize_model():
    # Load pre-trained GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

def generate_response(tokenizer, model, input_text, max_length=50):
    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate a response
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)

    # Decode and return the response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def main():
    print("Initializing AI chatbot...")
    tokenizer, model = initialize_model()

    print("Chatbot is ready. Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        response = generate_response(tokenizer, model, user_input)
        print("AI: " + response)

if _name_ == "_main_":
    main()
