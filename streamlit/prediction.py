import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from huggingface_hub import hf_hub_download

trained_model = None
class SentimentClassifier(nn.Module):
    def __init__(self):
        super().__init__() #Inheritting from Module Class
        self.bert = BertModel.from_pretrained('bert-base-uncased') #Bert Model to generate contextual embedding

        for param in self.bert.parameters():
            param.requires_grad = False  #Freezing parameters of bert model

        self.classifier = nn.Sequential(   #Implementing a seperate fnn classifier
            nn.Linear(self.bert.config.hidden_size,256), # Bert produces embeddings of size hidden_size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self,input_ids,attention_mask):
        bert_output = self.bert(input_ids = input_ids,attention_mask = attention_mask) #Passing input id and attention mask to bert model
        sentence_embedding = bert_output.last_hidden_state[:,0,:] # Accesing CLS (semantic meaning of entire sentence)
        return self.classifier(sentence_embedding) #Passing CLS to classifier

def tokenize(review):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoding = tokenizer(review,padding=True,truncation=True,return_tensors='pt')
    input_ids,attention_mask = encoding['input_ids'],encoding['attention_mask']
    return input_ids, attention_mask

def predict(review):
    input_ids,attention_mask = tokenize(review)

    global trained_model
    if trained_model is None:
        model_path = hf_hub_download(repo_id="itsmoksh/sentiment-bert-model", filename="sentiment_model.pth",map_location=torch.device('cpu'))
        trained_model= SentimentClassifier()
        trained_model.load_state_dict(torch.load(model_path))
        trained_model.eval()
    with torch.no_grad():
        output = trained_model(input_ids = input_ids,attention_mask = attention_mask)
        result = "Positive Review" if output>0.5 else "Negative Review"
        proba = output.item() if result=="Positive Review" else 1-output.item()
        return result,proba

