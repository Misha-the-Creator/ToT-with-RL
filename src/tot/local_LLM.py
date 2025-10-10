from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional
import torch

class Liquid_LLM:
    def __init__(self, model_id="LiquidAI/LFM2-2.6B", 
                 device_map='auto',
                 dtype=torch.bfloat16):
        
        self.model_id = model_id
        self.device_map = device_map
        self.dtype = dtype
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=self.device_map,
            dtype=self.dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, 
                prompt: str, 
                temperature: float = 0.7, 
                max_tokens: int = 1000, 
                n: int = 1,
                stop: Optional[List[str]] = None,
                **kwargs) -> List[str]:
        print("=== PROMPT ===")
        print(prompt)
        print("==============")
        outputs = []
        
        for _ in range(n):
            messages = [{"role": "user", "content": prompt}]
            
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=True,
            ).to(self.model.device)
            
            generation_config = {
                "do_sample": True,
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "repetition_penalty": 1.05,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            generation_config.update(kwargs)
            
            with torch.no_grad():
                output = self.model.generate(input_ids, **generation_config)
            
            generated_tokens = output[0][input_ids.shape[1]:]
            decoded_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            if stop:
                for stop_word in stop:
                    if stop_word in decoded_output:
                        decoded_output = decoded_output.split(stop_word)[0]
            
            outputs.append(decoded_output.strip())
        print("=== RESPONSE ===")
        print(str(outputs))
        print("================")
        return outputs

    def chat(self, 
             messages: List[dict], 
             temperature: float = 0.3, 
             max_tokens: int = 1000, 
             n: int = 1,
             stop: Optional[List[str]] = None,
             **kwargs) -> List[str]:
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to(self.model.device)
        
        outputs = []
        
        for _ in range(n):
            generation_config = {
                "do_sample": True,
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "repetition_penalty": 1.05,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            generation_config.update(kwargs)
            
            with torch.no_grad():
                output = self.model.generate(input_ids, **generation_config)
            
            generated_tokens = output[0][input_ids.shape[1]:]
            decoded_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            if stop:
                for stop_word in stop:
                    if stop_word in decoded_output:
                        decoded_output = decoded_output.split(stop_word)[0]
            
            outputs.append(decoded_output.strip())
        
        return outputs

# Функции для совместимости
def gpt(prompt, model=None, temperature=0.7, max_tokens=1000, n=1, stop=None, **kwargs) -> list:
    llm = Liquid_LLM()
    return llm.generate(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        stop=stop,
        **kwargs
    )

def chatgpt(messages, model=None, temperature=0.7, max_tokens=1000, n=1, stop=None, **kwargs) -> list:
    llm = Liquid_LLM()
    return llm.chat(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        stop=stop,
        **kwargs
    )

def gpt_usage(backend="local"):
    return {
        "completion_tokens": 0, 
        "prompt_tokens": 0, 
        "cost": 0,
        "backend": backend
    }