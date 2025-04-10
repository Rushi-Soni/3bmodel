"""
TurboTalk Inference Script
Provides interactive chat interface with continuous learning capabilities
"""

import os
import torch
import logging
from tokenizers import Tokenizer
from t.t.train_v2 import TurbotalkModel, TrainingConfig, self_learn, save_self_learning_data, load_self_learning_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("turbotalk_inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer():
    """Load the model and tokenizer with error handling"""
    try:
        config = TrainingConfig()
        
        # Load tokenizer
        tokenizer_path = "turbotalk_tokenizer.json"
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path)
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<PAD>"), pad_token="<PAD>")
        tokenizer.enable_truncation(max_length=config.max_seq_len)
        logger.info("Tokenizer loaded successfully")
        
        # Load model
        model_path = "turbotalk_checkpoints_v2/final_model.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        model = TurbotalkModel(config)
        model.load_state_dict(torch.load(model_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully and moved to {device}")
        
        return model, tokenizer, config, device
    except Exception as e:
        logger.error(f"Error loading model and tokenizer: {str(e)}")
        raise

def generate_response(model, tokenizer, prompt, config, device):
    """Generate response with error handling"""
    try:
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
        
        # Select random personality trait
        trait_idx = torch.randint(0, len(config.personality_traits), (1,), device=device)
        trait = config.personality_traits[trait_idx.item()]
        logger.debug(f"Using personality trait: {trait}")
        
        # Generate with reasoning
        output_ids = model.generate(
            input_ids,
            max_length=100,
            reasoning_steps=config.reasoning_steps,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            trait_idx=trait_idx
        )
        
        response = tokenizer.decode(output_ids[0])
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while generating a response."

def inference_loop():
    """Main inference loop with continuous learning"""
    try:
        # Load model and tokenizer
        model, tokenizer, config, device = load_model_and_tokenizer()
        
        # Load existing self-learning data
        self_learning_data = load_self_learning_data()
        logger.info(f"Loaded {len(self_learning_data)} previous interactions")
        
        step = 0
        print("\nTurboTalk AI Initialized. Type 'exit' to end the conversation.\n")
        
        while True:
            # Get user input
            try:
                prompt = input("\nYou: ").strip()
                if not prompt:
                    print("Please enter a message.")
                    continue
                if prompt.lower() == "exit":
                    print("\nThank you for chatting! Goodbye!")
                    break
                
                # Generate and display response
                response = generate_response(model, tokenizer, prompt, config, device)
                print(f"\nTurboTalk: {response}")
                
                # Update self-learning data
                self_learning_data.append(f"Prompt: {prompt}\nResponse: {response}")
                step += 1
                
                # Periodic self-learning
                if step % 10 == 0:
                    logger.info("Performing self-learning update...")
                    self_learn(model, tokenizer, config, self_learning_data[-10:])
                    save_self_learning_data(self_learning_data)
                    logger.info("Model updated with recent interactions")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Saving data and exiting...")
                break
            except Exception as e:
                logger.error(f"Error in conversation loop: {str(e)}")
                print("\nI encountered an error. Let's continue our conversation.")
        
        # Final save of self-learning data
        save_self_learning_data(self_learning_data)
        logger.info("Inference session ended, self-learning data saved")
        
    except Exception as e:
        logger.error(f"Fatal error in inference loop: {str(e)}")
        print("\nI apologize, but I encountered a critical error and need to shut down.")

if __name__ == "__main__":
    inference_loop() 