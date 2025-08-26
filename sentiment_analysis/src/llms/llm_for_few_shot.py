import json
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from peft import LoraConfig
from sentiment_analysis.src.llms.token_id import get_token
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, Trainer, set_seed
import copy
    
MODEL_ID = {
    'Llama3.1-I' : 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'Llama3.1'   : 'meta-llama/Meta-Llama-3.1-8B'
}

SEED = 2024

set_seed(SEED)

class LLM():
    def __init__(
                    self, 
                    llm_method: str = 'Llama3-I',
                    prompt_dir = "",
                ):
        
        self.llm_method = llm_method
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = MODEL_ID[self.llm_method]
        self.prompt_dir = prompt_dir
    
    def set_model(self, texts_for_few_shot):
        self.get_prompt_(texts_for_few_shot)

        self.token_access = get_token()

        self.model = AutoModelForCausalLM.from_pretrained(  self.model_name, 
                                                            torch_dtype="auto", 
                                                            device_map=self.device, 
                                                            offload_buffers=True, 
                                                            token=self.token_access, 
                                                            trust_remote_code=True,
                                                            use_cache=False#,
                                                            #load_in_8bit=True
                                                            #load_in_4bit=True
                                                            )
        
        self.tokenizer = AutoTokenizer.from_pretrained( self.model_name, 
                                                        torch_dtype="auto", 
                                                        device_map=self.device, 
                                                        offload_buffers=True, 
                                                        token=self.token_access, 
                                                        use_safetensors=True, 
                                                        trust_remote_code=True) 
                                                        #use_cache=False)
         
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]          

    def create_text_prompt(self):
        prompt = [
                    {
                    'role': 'system', 
                    'content':  self.system_prompt
                    }
                ]
        
        for comment in self.texts_for_few_shot.values():
            prompt += [ 
                    {
                    'role': 'user', 
                    'content': f'Input: {comment["text"]}:'
                    },
                    {
                    'role': 'assistant', 
                    'content': f'{comment["label"]}'
                    }
                ]

        return prompt

    def get_prompt_(self, texts_for_few_shot):
        with open(self.prompt_dir, 'r') as f:
            data = json.load(f)
        
        self.system_prompt = data["system_prompt"]
        self.categories = data["categories"]  
        self.texts_for_few_shot = texts_for_few_shot
        self.prompt = self.create_text_prompt()
        self.max_new_tokens = max([len(category.split()) for category in self.categories])
            
    def remove_tokens_for_classification(self, text, total_number_of_tokens, target_number_of_tokens=5000):
        words = text.split()
        tokens_removed = 0
        current_number_of_tokens = total_number_of_tokens
        
        if current_number_of_tokens > target_number_of_tokens*2:
            step_size = len(words) // 2
        else:
            step_size = len(words) // 20
                
        while total_number_of_tokens - tokens_removed > target_number_of_tokens:
            words = words[:-step_size]
            truncated_text = " ".join(words)
            
            prompt = self.create_text_prompt({"text": truncated_text}, predict=True)

            inputs = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt", return_dict=True)

            current_number_of_tokens = inputs['input_ids'][0].numel()
            tokens_removed = total_number_of_tokens - current_number_of_tokens
            
            if current_number_of_tokens > target_number_of_tokens*2:
                step_size = len(words) // 2
            else:
                step_size = len(words) // 20
        
        return inputs.to("cuda")

    def add_text_in_prompt_to_classify(self, text):
        prompt = copy.deepcopy(self.prompt)
        prompt += [ 
                    {
                    'role': 'user', 
                    'content': f'{text}'
                    }
                ]
        return prompt
    
    def predict_llm_(self, text):
        
        default_prompt = self.add_text_in_prompt_to_classify(text)
                        
        inputs = self.tokenizer.apply_chat_template(default_prompt, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
        if inputs['input_ids'].numel() > 5000:
            print('Removing text', inputs['input_ids'].numel(), end=' ')
            inputs = self.remove_tokens_for_classification(text=text, total_number_of_tokens=inputs['input_ids'].numel()) 
            print(inputs['input_ids'].numel())
        
        outputs = self.model.generate(inputs['input_ids'], 
                                    attention_mask = inputs['attention_mask'],  
                                    max_new_tokens=self.max_new_tokens+5,
                                    eos_token_id=self.terminators,
                                    pad_token_id=self.tokenizer.eos_token_id,
                                    do_sample=True,
                                    temperature=0.1,
                                    top_p=0.9,
                                    use_cache=False)
        response_model = outputs[0][inputs['input_ids'].shape[-1]:]
        response_model = self.tokenizer.decode(response_model, skip_special_tokens=True)
        response_model = response_model.lower()

        return response_model

    def predict(self, data):
        print(self.llm_method)
        
        y_text = []

        X = data['comments'].tolist()

        for index, text in enumerate(tqdm(X, desc="Predict", ascii=True)):

            if index in self.texts_for_few_shot.values():
                y_text.append(self.texts_for_few_shot[index]['label'])
                continue

            response_model = self.predict_llm_(f'Input: {text}')

            while response_model not in self.categories:
                print('Regenerating response.')
                new_text = f'Attention! Classify only into the categories you were instructed to.\nInput: {text}'
                response_model = self.predict_llm_(new_text)

            y_text.append(response_model)

            torch.cuda.empty_cache()
        
        return y_text

if __name__ == '__main__':

    print("Main")