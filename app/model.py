from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class HuggingFaceModel:
    def __init__(self, model_name="HuggingFaceM4/tiny-random-LlamaForCausalLM"):
        """
        Initialize the Hugging Face model.

        Args:
            model_name (str): Name of the Hugging Face model to use
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.initialized = False

    def load_model(self):
        """Load the model and tokenizer from Hugging Face"""
        try:
            print(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            self.initialized = True
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def generate_response(self, prompt, max_length=100):
        """
        Generate a response based on the input prompt.

        Args:
            prompt (str): The input prompt
            max_length (int): Maximum length of the generated response

        Returns:
            str: The generated response
        """
        if not self.initialized:
            success = self.load_model()
            if not success:
                return "Failed to load the model. Please try again."

        try:
            # Tokenize the input prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )

            # Decode and return the response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the new part (response)
            response = generated_text[len(prompt) :].strip()
            return response if response else "I don't have a response for that."

        except Exception as e:
            return f"Error generating response: {e}"
