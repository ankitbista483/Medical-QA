import pandas as pd
from transformers import BartTokenizer

class QATokenizer:
    def __init__(self, data_path='/Users/ankitbista/Desktop/practice/MedQuAD/Medical-QA/data.csv',
                 model_name='facebook/bart-large', max_len=1024):
        self.data_path = data_path
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.max_len = max_len

    def tokenized_data(self):
        df = pd.read_csv(self.data_path)
        tokenized_inputs = []

        for _, row in df.iterrows():
            question = row['Question']
            answer = row['Answer']

           
            combined_text = f"{question} </s> {answer}"  

            encoding = self.tokenizer(
                combined_text,
                truncation=True,  
                padding='max_length',
                max_length=self.max_len,
                return_tensors='pt'
            )
            encoding['labels'] = encoding['input_ids'].clone() 
            tokenized_inputs.append(encoding)

        return tokenized_inputs
