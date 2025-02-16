from transformers import BertTokenizer
import pandas as pd


class QATokenizer:
    def __init__(self,model_name = 'bert-large-uncased',data_path = None):
        self.tokenized_input  = []
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.df= pd.read_csv(data_path)

    def tokenized_data(self):
        
        for _,row in self.df.iterrows():
            question = row['Question']
            answer = row['Answer']

            tokenized_input = self.tokenizer(
                question,answer, 
                padding = True, 
                truncation = True, 
                return_tensors = 'pt')

            return self.tokenized_input.append(tokenized_input)