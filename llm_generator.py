from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import List


class LLMGenerator:
    """Handles text generation using a lightweight Hugging Face model."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """
        Initialize the language model.
        
        Args:
            model_name: Name of the Hugging Face model
                       Default: microsoft/DialoGPT-medium (lightweight, good for conversation)
                       Alternative: "distilgpt2" (even lighter)
        """
        print(f"Loading language model: {model_name}")
        
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create pipeline for easier text generation
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            do_sample=True,
            temperature=0.7,
            max_length=512,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        print(f"Language model loaded successfully")
    
    def generate_response(self, query: str, context_chunks: List[str], max_length: int = 300) -> str:
        """
        Generate a response based on query and retrieved context.
        
        Args:
            query: User query
            context_chunks: List of retrieved text chunks
            max_length: Maximum length of generated response
            
        Returns:
            Generated response text
        """
        # Prepare context
        context = "\n\n".join(context_chunks)
        
        # Create prompt
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate response
        try:
            generated = self.generator(
                prompt,
                max_length=len(prompt.split()) + max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract the generated text (remove the prompt)
            full_response = generated[0]['generated_text']
            response = full_response[len(prompt):].strip()
            
            # Clean up the response
            if response.startswith("Answer:"):
                response = response[7:].strip()
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"I'm sorry, I encountered an error while generating a response. Please try again."
    
    def generate_simple_response(self, prompt: str, max_length: int = 150) -> str:
        """
        Generate a simple response without context formatting.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated response
            
        Returns:
            Generated response text
        """
        try:
            generated = self.generator(
                prompt,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = generated[0]['generated_text'][len(prompt):].strip()
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while generating a response."
