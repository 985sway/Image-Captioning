import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed_size=embed_size
        self.hidden_size=hidden_size
        self.vocab_size=vocab_size
        self.num_layers=num_layers
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        captions=self.embed(captions[:,:-1])
        inputs=torch.cat((features.view(len(features),1,self.embed_size),captions),1)
        inputs = Variable(inputs.cuda())
        outputs,hidden=self.lstm(inputs)
        outputs=self.fc(outputs)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
# " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        ids=[]
        for i in range(max_len):
            outputs, states = self.lstm(inputs, states)
            outputs = self.fc(outputs.view(1,-1))
            pred = outputs.argmax(1)
            ids.append(pred.item())
            inputs = self.embed(pred)
            inputs = inputs.view(1,1,-1)
        return ids
    
























