from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Define the local model path
model_path = r"D:\ttm\model\3bmodel\t\M\TTM\model\snapshots\7bcac572ce56db69c1ea7c8af255c5d7c9672fc2"

# Load the model and tokenizer from local path
print("Loading model and tokenizer...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Model and tokenizer loaded successfully!")

# Create a text generation pipeline
pipe = pipeline(
    "text2text-generation", 
    model=model, 
    tokenizer=tokenizer
)

# Test the model
prompt = "Hello, how are you?"
print(f"\nTesting model with prompt: '{prompt}'")
output = pipe(prompt, max_length=50)
print("Output:", output[0]['generated_text'])

# Try another example
prompt2 = "What is python snake??"
print(f"\nTesting with prompt: '{prompt2}'")
output2 = pipe(prompt2, max_length=50)
print("Output:", output2[0]['generated_text'])

print("\nModel is working correctly!")