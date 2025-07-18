import torch
import torch.nn as nn
import torch.nn.functional as F

class FastText(nn.Module):
    def __init__(self, embedding_pretrained, n_vocab, embed, dropout, hidden_size, num_classes, n_gram_vocab):
        super(FastText, self).__init__()
        if embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(n_vocab, embed, padding_idx=n_vocab - 1)
        self.embedding_ngram2 = nn.Embedding(n_gram_vocab, embed)
        self.embedding_ngram3 = nn.Embedding(n_gram_vocab, embed)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed * 3, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out_word = self.embedding(x[0])
        out_bigram = self.embedding_ngram2(x[2])
        out_trigram = self.embedding_ngram3(x[3])
        out = torch.cat((out_word, out_bigram, out_trigram), -1)
        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
