import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeuralParser(nn.Module):
    def __init__(self, len_word_embed, len_tag_embed, word2id, tag2id, label2id, vocab2id, pretrained_word_embeddings,
                 pretrained_tag_embeddings):
        super(NeuralParser, self).__init__()

        self.word2id = word2id
        self.tag2id = tag2id
        self.vocab2id = vocab2id

        self.uni_lstm_hidden_size = 200
        self.lstm_hidden_size = 400

        self.word_embeddings = nn.Embedding(len(pretrained_word_embeddings), len_word_embed)
        self.word_embeddings.weight = nn.Parameter(pretrained_word_embeddings)

        self.tag_embeddings = nn.Embedding(len(pretrained_tag_embeddings), len_tag_embed)
        self.tag_embeddings.weight = nn.Parameter(pretrained_tag_embeddings)

        self.char_embeddings = nn.Embedding(len(vocab2id), self.uni_lstm_hidden_size)

        self.uni_lstm = nn.LSTM(input_size=self.uni_lstm_hidden_size, hidden_size=self.uni_lstm_hidden_size)

        self.mlp_char_layer1 = nn.Linear(in_features=self.uni_lstm_hidden_size,
                                         out_features=self.uni_lstm_hidden_size * 2)
        self.mlp_char_layer2 = nn.Linear(in_features=self.uni_lstm_hidden_size * 2, out_features=len_word_embed)

        self.bi_lstm = nn.LSTM(input_size=len_word_embed + len_tag_embed,
                               hidden_size=self.lstm_hidden_size,
                               dropout=0.33,
                               bidirectional=True, num_layers=3)

        # score head
        self.mlp_arc_head1 = nn.Linear(in_features=self.lstm_hidden_size * 2, out_features=500)
        self.mlp_arc_head2 = nn.Linear(in_features=500, out_features=20)

        # score dependent
        self.mlp_arc_dep1 = nn.Linear(in_features=self.lstm_hidden_size * 2, out_features=500)
        self.mlp_arc_dep2 = nn.Linear(in_features=500, out_features=20)

        # score label from head and to dependent
        self.mlp_label_head1 = nn.Linear(in_features=self.lstm_hidden_size * 2, out_features=200)
        self.mlp_label_head2 = nn.Linear(in_features=200, out_features=20)

        self.mlp_label_dep1 = nn.Linear(in_features=self.lstm_hidden_size * 2, out_features=200)
        self.mlp_label_dep2 = nn.Linear(in_features=200, out_features=20)

        # choose most appropriate label
        self.mlp_classifier1 = nn.Linear(in_features=20 * 2, out_features=60)
        self.mlp_classifier2 = nn.Linear(in_features=60, out_features=len(label2id))

        self.U1 = nn.Parameter(torch.randn(20, 20))
        self.u2 = nn.Parameter(torch.randn(1, 20))

    def MLP_char(self, h):
        return self.mlp_char_layer2(torch.nn.functional.leaky_relu(self.mlp_char_layer1(h)))

    def MLP_arc_head(self, h):
        return self.mlp_arc_head2(torch.nn.functional.relu(self.mlp_arc_head1(h)))

    def MLP_arc_dep(self, h):
        return self.mlp_arc_dep2(torch.nn.functional.relu(self.mlp_arc_dep1(h)))

    def MLP_label_head(self, h):
        return self.mlp_label_head2(torch.nn.functional.relu(self.mlp_label_head1(h)))

    def MLP_label_dep(self, h):
        return self.mlp_label_dep2(torch.nn.functional.relu(self.mlp_label_dep1(h)))

    def MLP_classifier(self, h):
        return self.mlp_classifier2(torch.nn.functional.relu(self.mlp_classifier1(h)))

    def init_hidden(self):
        return torch.zeros((2 * 3, self.lstm_hidden_size)).to(device), torch.zeros((2 * 3, self.lstm_hidden_size)).to(
            device)

    def init_hidden2(self):
        return torch.zeros((1, self.uni_lstm_hidden_size)).to(device), torch.zeros((1, self.uni_lstm_hidden_size)).to(
            device)

    def forward(self, sequence):
        words, tags = [], []

        h, c = self.init_hidden2()
        emb_chars = []
        for i in range(len(sequence)):
            chars = []
            if i == 0:
                sequence[i].form = "<ROOT>"
                sequence[i].upos = "<ROOT>"
            if sequence[i].form not in self.word2id:
                sequence[i].form = "unk"
            if sequence[i].upos not in self.tag2id:
                sequence[i].upos = "unk"

            if sequence[i].form == "<ROOT>":
                chars.append(self.vocab2id["<ROOT>"])
            else:
                for let in sequence[i].form:
                    if let not in self.vocab2id:
                        let = "unk"
                    chars.append(self.vocab2id[let])

            if sequence[i].feat is not None and sequence[i].feat != "_":
                for feat in sequence[i].feat.split("|"):
                    feat = feat[feat.index("=") + 1:].strip()
                    if feat in self.vocab2id:
                        chars.append(self.vocab2id[feat])
                    else:
                        chars.append(self.vocab2id["unk"])

            emb = self.char_embeddings(torch.tensor(chars).to(device))
            _, (h, c) = self.uni_lstm(emb, (h, c))
            h_char = self.MLP_char(h.to(device))
            emb_chars.append(h_char)

            words.append(self.word2id[sequence[i].form])
            tags.append(self.tag2id[sequence[i].upos])

        words, tags = torch.tensor(words).to(device), torch.tensor(tags).to(device)

        emb_chars = torch.stack([emb_chars[i] for i in range(len(emb_chars))], dim=0).squeeze(dim=1).to(device)

        emb_words = self.word_embeddings(words)

        # element-wise addition of the representation of whole word
        # and representations of consisting characters + morphological features
        emb_words += emb_chars

        emb_tags = self.tag_embeddings(tags)

        xi = torch.cat([emb_words, emb_tags], dim=1).to(device)

        h1, c1 = self.init_hidden()

        o2, _ = self.bi_lstm(xi, (h1, c1))
        o2 = o2.to(device)

        h_arc_head = self.MLP_arc_head(o2)
        h_arc_dep = self.MLP_arc_dep(o2)

        # arc predictions
        adj_matrix = h_arc_head @ self.U1.to(device) @ torch.t(h_arc_dep) + h_arc_head @ torch.t(self.u2.to(device))

        # label predictions
        label_head = self.MLP_label_head(o2)
        label_dep = self.MLP_label_dep(o2)

        h_label = torch.cat([label_head, label_dep], dim=1).to(device)

        labels = self.MLP_classifier(h_label)

        return torch.t(adj_matrix).to(device), labels.to(device)
