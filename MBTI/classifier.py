import torch
import torch.nn as nn
import torch.nn.functional as F

# MLP
class MLPClassifier(nn.Module):
    """ MBTI Binary Classifier """
    def __init__(self,
                 base_model,
                 hidden_dim,
                 num_classes = 2,
                 dropout = 0.2):
        super(MLPClassifier, self).__init__()

        assert type(hidden_dim) == list, ValueError("hidden_dim should be list type")

        self.base_model   = base_model
        self.num_classes  = num_classes      # number of classes
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        emb_dim = 2
        self.embedding = nn.Embedding(num_embeddings=60, embedding_dim=emb_dim)
        self.dimensions = [1024 + emb_dim + 5] + hidden_dim + [self.num_classes]        # dimension list for all layers 
        # layer stacks
        self.layers = nn.ModuleList(
            [nn.Linear(self.dimensions[i - 1], self.dimensions[i]) for i in range(1, len(self.dimensions))])
        self.activaiton = nn.LeakyReLU()

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight.data, gain=1.0)

    def forward(self,
                input_ids,
                attention_mask,
                others,
                q_num):

        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        outputs = self.dropout(outputs[1])    # Put last hidden state into dropout layer : [batch_size, 768]
        q_embedded = self.embedding(q_num)    # [batch_size, 2]
        others = others.to(torch.float32)     # [batch_size, 5] 
        # gender = gender.view(-1, 1).to(torch.float32)                # Preprocess
        # age = age.view(-1, 1).to(torch.float32)                      # Preprocess
        outputs = torch.cat((q_embedded, outputs, others), 1)          # [batch_size, emb_dim + 1024 + 5]

        for _, layer in enumerate(self.layers):
            outputs = layer(outputs)
            outputs = self.activaiton(outputs)

        return outputs