import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

class QATokenizer:
    def __init__(self, data_path='/Users/ankitbista/Desktop/practice/MedQuAD/Medical-QA/data.csv', model_name='facebook/bart-large', max_len=1024):
        self.data_path = data_path
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.max_len = max_len

    def tokenized_data(self):
        self.df = pd.read_csv(self.data_path)

        tokenized_inputs = []
        for _, row in self.df.iterrows():
            question = row['Question']
            answer = row['Answer']
            
            context = answer
            encoding = self.tokenizer(
                question,
                context,
                truncation='only_second',  
                padding='max_length',
                max_length=self.max_len,
                return_tensors='pt'
            )
            
            answer_start = len(self.tokenizer.encode(question, add_special_tokens=True))
            answer_end = len(encoding['input_ids'][0]) - 1 

            encoding['start_positions'] = torch.tensor([answer_start])
            encoding['end_positions'] = torch.tensor([answer_end])
            
            tokenized_inputs.append(encoding)

        return tokenized_inputs



