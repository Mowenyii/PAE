from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import torch
import numpy as np
import random

import random
import pdb

class SFT_Datasets(Dataset):
    def __init__(self, device="cuda",block_size=77) -> None:
        super().__init__()
        print("Load Dataset.")
        self.device=device
        tokenizer=GPT2Tokenizer.from_pretrained("gpt2",device=device)
        tokenizer.pad_token = tokenizer.eos_token

        self.block_size = block_size

        def replace_period_with_comma(text):
            replaced_text = text
            replaced_text_list = list(replaced_text)
            for i, char in enumerate(replaced_text_list):
                if char == ',':
                    if random.random() < 0.5:
                        replaced_text_list[i] = '.'
            final_text = ''.join(replaced_text_list)
            return final_text

        self.tokens=[]

        prompt_list=np.load("train_data.npy")
        for prompt in prompt_list:
            if random.random() < 0.5: 
                first_term=prompt
                if first_term.isupper():  
                    first_term = first_term.lower()  
                else:
                    first_term = first_term.capitalize()
                response_text=first_term
            else:
                response_text=prompt
            if "<|endoftext|>" in response_text:
                response = tokenizer(replace_period_with_comma(response_text))
            else:
                response = tokenizer(replace_period_with_comma(response_text) + "<|endoftext|>")
            self.tokens+=response['input_ids']

        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        

    def __len__(self):
        import sys
        return sys.maxsize

    def __getitem__(self, idx):
        start = random.randint(0, len(self.tokens) - self.block_size - 2)
        x = self.tokens[start:start + self.block_size]
        y = self.tokens[start + 1:start + self.block_size + 1]
        return x, y



class PPO_Dataset(Dataset):
    def __init__(self, device="cuda",block_size=77) -> None:
        super().__init__()

        self.device=device
        tokenizer=GPT2Tokenizer.from_pretrained("gpt2",device=device)
        tokenizer.pad_token = tokenizer.eos_token
        self.block_size = block_size


        def replace_period_with_comma(text):
            replaced_text = text
            replaced_text_list = list(replaced_text)
            for i, char in enumerate(replaced_text_list):
                if char == ',':
                    if random.random() < 0.5:
                        replaced_text_list[i] = '.'
            final_text = ''.join(replaced_text_list)
            return final_text

        self.tokens = []

        prompt_list=np.load("train_data.npy")
        for prompt in prompt_list:
            if random.random() < 0.5: 
                first_term=prompt
                if first_term.isupper():  
                    first_term = first_term.lower()  
                else:
                    first_term = first_term.capitalize()
                response_text=first_term
            else:
                response_text=prompt
            if "<|endoftext|>" in response_text:
                tokens = tokenizer(replace_period_with_comma(response_text), 
                        max_length=77,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt")
            else:
                tokens = tokenizer(replace_period_with_comma(response_text) + "<|endoftext|>", 
                        max_length=77,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt")

            self.tokens.append(
            [tokens['input_ids'], tokens['attention_mask'], torch.sum(tokens['attention_mask'])])

        
    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx][0], self.tokens[idx][1], self.tokens[idx][2]  





