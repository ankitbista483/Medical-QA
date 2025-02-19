import torch
from transformers import BartForConditionalGeneration, BartTokenizer

class QAModel:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def generate_answer(self, question, max_length=50):
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

# Usage
model_path = '/Users/ankitbista/Desktop/practice/MedQuAD/Medical-QA/best_model.pth'
qa_model = QAModel(model_path)

question = "What is cell lung cancer?"
answer = qa_model.generate_answer(question)
print(f"Question: {question}")
print(f"Answer: {answer}")
