from Data.tokenizer_script import QATokenizer

class QADataset:
    def __init__(self):
        self.token = QATokenizer()
        

    def tokenized_qa(self):
        return self.token.tokenized_data()
        
    
    def __len__(self):
        return len(self.tokenized_qa())

    def __getitem__(self,idx):
        return self.tokenized_qa()[idx]



