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
            
            print(tokenized_input)

            return self.tokenized_input.append(tokenized_input)
        


j = QATokenizer(data_path='/Users/ankitbista/Desktop/practice/MedQuAD/Medical-QA/data.csv')
j = j.tokenized_data()