import torch
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, AdamW, BartTokenizer
from Data.QADataset import QADataset
from tqdm import tqdm

class Model:
    def __init__(self, model_name='facebook/bart-base', optimizer=AdamW, lr=2e-5, batch_size=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qa_dataset = QADataset()
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.optimizer = optimizer(self.model.parameters(), lr=lr,no_deprecation_warning=True)
        self.batch_size = batch_size

    def data_loader(self):
        dataloader = DataLoader(self.qa_dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader
        
    
    import torch
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, AdamW, BartTokenizer
from Data.QADataset import QADataset
from tqdm import tqdm

class Model:
    def __init__(self, model_name='facebook/bart-base', optimizer=AdamW, lr=2e-5, batch_size=1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qa_dataset = QADataset()
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.optimizer = optimizer(self.model.parameters(), lr=lr,no_deprecation_warning=True)
        self.batch_size = batch_size

    def data_loader(self):
        dataloader = DataLoader(self.qa_dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader
        
    
    def model_trainer(self, epochs=5, accumulation_steps=4, patience=3):
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0
            
            for i, batch in enumerate(self.data_loader()):
                input_ids = batch['input_ids'].squeeze(1).to(self.device)
                attention_mask = batch['attention_mask'].squeeze(1).to(self.device)
                labels = input_ids.clone()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / accumulation_steps

                loss.backward()

                if (i + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item() * accumulation_steps
                batch_count += 1

                if i % 10 == 0:
                    print(f"Epoch {epoch + 1} Step {i} Loss: {loss.item() * accumulation_steps}")




    def save_model(self, save_path = '/Users/ankitbista/Desktop/practice/MedQuAD/Medical-QA/qa_model.pth'):
        torch.save(self.model.state_dict(), save_path)


    def generate_answer(self, question, max_length=50):
        self.model.eval()
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer


if __name__ == "__main__":
    m = Model()
    m.model_trainer()

    m.save_model('qa_model.pth')
    question = "What is cell lung cancer?"
    answer = m.generate_answer(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")



    def save_model(self, save_path = '/Users/ankitbista/Desktop/practice/MedQuAD/Medical-QA/qa_model.pth'):
        torch.save(self.model.state_dict(), save_path)


    def generate_answer(self, question, max_length=50):
        self.model.eval()
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return answer


# if __name__ == "__main__":
#     m = Model()
#     m.model_trainer()

#     m.save_model('/Users/ankitbista/Desktop/practice/MedQuAD/Medical-QA/best_model.pth')
#     question = "What is cell lung cancer?"
#     answer = m.generate_answer(question)
#     print(f"Question: {question}")
#     print(f"Answer: {answer}")

m = Model()  # Create an instance of your Model class
m.model.load_state_dict(torch.load('/Users/ankitbista/Desktop/practice/MedQuAD/Medical-QA/best_model.pth'))
m.model.eval()  # Set the model to evaluation mode
question = "What is cell lung cancer?"
answer = m.generate_answer(question)
print(f"Question: {question}")
print(f"Answer: {answer}")
