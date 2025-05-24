import requests
from bs4 import BeautifulSoup
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, BartForConditionalGeneration, pipeline
import argparse
import time
import sys
import os

class TurboTalkAI:
    def __init__(self, model_path):
        print(f"Loading TurboTalk AI model from {model_path}...")
        
        # Check if the model directory exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
            
        # Try to determine the base model architecture from config.json
        try:
            # First try to load with AutoTokenizer which is more forgiving
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Try different model architectures
            try:
                # Try T5 architecture
                self.model = T5ForConditionalGeneration.from_pretrained(model_path)
                print("Loaded model using T5ForConditionalGeneration architecture")
            except:
                try:
                    # Try BART architecture
                    self.model = BartForConditionalGeneration.from_pretrained(model_path)
                    print("Loaded model using BartForConditionalGeneration architecture")
                except:
                    # Fall back to forcing AutoModel to ignore architecture type
                    print("Attempting to load with trust_remote_code=True and ignore_mismatched_sizes=True...")
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_path, 
                        trust_remote_code=True,
                        ignore_mismatched_sizes=True
                    )
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nTrying alternative loading method...")
            
            # If all else fails, try to modify the config.json to use a standard architecture
            try:
                import json
                config_path = os.path.join(model_path, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Save original model_type for reference
                    original_type = config.get("model_type", "unknown")
                    print(f"Original model_type: {original_type}")
                    
                    # Try to determine appropriate architecture based on config keys
                    if "d_model" in config and "num_layers" in config:
                        # Likely T5-like architecture
                        config["model_type"] = "t5"
                    elif "hidden_size" in config and "encoder_layers" in config:
                        # Likely BART-like architecture
                        config["model_type"] = "bart"
                    else:
                        # Default to T5
                        config["model_type"] = "t5"
                    
                    print(f"Setting model_type to: {config['model_type']}")
                    
                    # Save modified config
                    temp_config_path = os.path.join(model_path, "config_temp.json")
                    with open(temp_config_path, 'w') as f:
                        json.dump(config, f)
                    
                    # Rename files to preserve original
                    os.rename(config_path, os.path.join(model_path, "config_original.json"))
                    os.rename(temp_config_path, config_path)
                    
                    # Try loading again
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                    print(f"Successfully loaded model after changing model_type to {config['model_type']}")
                else:
                    raise FileNotFoundError(f"Config file not found at {config_path}")
            except Exception as e:
                print(f"All loading attempts failed: {e}")
                raise
        
        # Create the pipeline
        self.pipe = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)
        print("TurboTalk AI model loaded successfully!")
        
    def search_web(self, query):
        """Search the web for information based on the query."""
        print(f"Searching the web for: {query}")
        
        # Format the query for a search URL
        search_query = query.replace(' ', '+')
        url = f"https://www.google.com/search?q={search_query}"
        
        # Set a user agent to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            # Send the request
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract search results
            search_results = []
            
            # Get the main search results
            for result in soup.select('div.g'):
                title_element = result.select_one('h3')
                if title_element:
                    title = title_element.get_text()
                    
                    # Get the snippet/description
                    snippet_element = result.select_one('div.VwiC3b')
                    snippet = snippet_element.get_text() if snippet_element else ""
                    
                    # Get the URL
                    link_element = result.select_one('a')
                    link = link_element['href'] if link_element and 'href' in link_element.attrs else ""
                    
                    if title and snippet:
                        search_results.append({
                            'title': title,
                            'snippet': snippet,
                            'link': link
                        })
            
            # Extract the featured snippet if available
            featured_snippet = soup.select_one('div.IZ6rdc')
            if featured_snippet:
                featured_text = featured_snippet.get_text()
                search_results.insert(0, {
                    'title': 'Featured Snippet',
                    'snippet': featured_text,
                    'link': ''
                })
            
            # Combine the search results into a single text
            combined_text = ""
            for i, result in enumerate(search_results[:5]):  # Limit to top 5 results
                combined_text += f"Result {i+1}:\nTitle: {result['title']}\nSnippet: {result['snippet']}\n\n"
            
            return combined_text if combined_text else "No relevant information found on the web."
            
        except Exception as e:
            print(f"Error searching the web: {e}")
            return f"Error searching the web: {str(e)}"
    
    def get_search_query(self, user_prompt):
        """Use the model to determine what to search for based on the user prompt."""
        search_instruction = f"Based on this user question, what should I search on the web to find relevant information? User question: {user_prompt}"
        search_query = self.pipe(search_instruction, max_length=100)[0]['generated_text']
        return search_query
    
    def generate_response(self, user_prompt, web_info=None):
        """Generate a response based on the user prompt and optional web information."""
        if web_info:
            prompt = f"User question: {user_prompt}\n\nWeb search information:\n{web_info}\n\nPlease provide a helpful response based on this information."
        else:
            prompt = user_prompt
            
        response = self.pipe(prompt, max_length=512, min_length=50)[0]['generated_text']
        return response
    
    def process_query(self, user_prompt, web_mode=False):
        """Process a user query with or without web search."""
        if web_mode:
            print("\nProcessing with web mode ON:")
            print("Step 1: Determining what to search for...")
            search_query = self.get_search_query(user_prompt)
            print(f"Generated search query: {search_query}")
            
            print("Step 2: Searching the web...")
            web_info = self.search_web(search_query)
            print("Web search completed")
            
            print("Step 3: Generating response using web information...")
            response = self.generate_response(user_prompt, web_info)
        else:
            print("\nProcessing with web mode OFF (using model knowledge only)...")
            response = self.generate_response(user_prompt)
            
        return response

def main():
    parser = argparse.ArgumentParser(description='TurboTalk AI with web search capability')
    parser.add_argument('--model_path', type=str, default=r"D:\ttm\model\3bmodel\t\M\TTM\finetuned_model",
                        help='Path to the fine-tuned model')
    parser.add_argument('--web_mode', action='store_true', help='Enable web search mode')
    
    args = parser.parse_args()
    
    # Initialize the model
    try:
        turbotalk = TurboTalkAI(args.model_path)
        
        print("\n" + "="*50)
        print("TurboTalk AI by Rango Productions")
        print("="*50)
        print(f"Web mode: {'ON' if args.web_mode else 'OFF'}")
        print("Type 'exit' to quit or 'toggle web' to switch web mode")
        print("="*50)
        
        # Interactive loop
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() == 'exit':
                print("Goodbye! Thank you for using TurboTalk AI.")
                break
                
            if user_input.lower() == 'toggle web':
                args.web_mode = not args.web_mode
                print(f"Web mode: {'ON' if args.web_mode else 'OFF'}")
                continue
                
            # Process the query
            start_time = time.time()
            response = turbotalk.process_query(user_input, args.web_mode)
            end_time = time.time()
            
            # Display the response
            print("\nTurboTalk AI:", response)
            print(f"\n(Response generated in {end_time - start_time:.2f} seconds)")
    
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure the model path is correct")
        print("2. Try installing transformers from source: pip install git+https://github.com/huggingface/transformers.git")
        print("3. Check if your model was fine-tuned from a standard architecture (T5, BART, etc.)")
        print("4. Ensure you have all required dependencies installed")

if __name__ == "__main__":
    main()