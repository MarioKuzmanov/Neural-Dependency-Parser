import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from gensim.models import Word2Vec

from model import NeuralParser
from conllu import read_conllu


class Trainer(object):

    def __init__(self, train_file):

        self.train_file = train_file

        self.vocab = {'М', 'Ж', 'У', 'А', 'д', 'з', 'щ', 'Х', 'Ю', 'м', 'т', 'К', 'З', 'Р', 'в', 'е', 'ъ', 'р',
                      'н', 'Б',
                      'В', 'ю',
                      'П', 'Ш', 'ш', 'Й', 'ц', 'Ъ', 'Н', 'а', 'Щ', 'Л', 'Ч', 'И', 'с', 'Т', 'о', 'г', 'п', 'ь',
                      'Е', 'О',
                      'Ф', 'б',
                      'Ц', 'к', 'и', 'Д', 'Г', 'я', 'ж', 'л', 'ф', 'Ь', 'Я', 'х', 'С', 'у', 'й', 'ч', 'unk',
                      '<ROOT>'}

        self.len_word_embed = 100
        self.len_tag_embed = 20

        self.word2id, self.tag2id, self.label2id, self.vocab2id = None, None, None, None
        self.pretrained_word_embeddings, self.pretrained_tag_embeddings = None, None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lexicon(train_file)

    def lexicon(self, train_file):

        words, tags = [], []
        all_words, all_tags, all_labels = [], [], set()

        features_vocab = set()
        for line in read_conllu(filename=train_file):
            line_words, line_tags = [], []

            for i in range(len(line)):
                if i == 0:
                    line[i].form = "<ROOT>"
                    line[i].upos = "<ROOT>"
                    line[i].deprel = "root"
                line_words.append(line[i].form)
                line_tags.append(line[i].upos)
                all_labels.add(line[i].deprel)

                if line[i].feat is not None and line[i].feat != "_":
                    for feat in line[i].feat.split("|"):
                        features_vocab.add(feat[feat.index("=") + 1:].strip())

            tags.append(line_tags)
            words.append(line_words)

            all_words.extend(line_words), all_tags.extend(line_tags)

        all_words, all_tags = sorted(set(all_words)), sorted(set(all_tags))
        all_labels = sorted(all_labels)

        self.word2id, self.tag2id = {w: idx for idx, w in enumerate(all_words)}, {tag: idx for idx, tag in
                                                                                  enumerate(all_tags)}
        self.label2id = {label: idx for idx, label in enumerate(all_labels)}

        word_embeddings_gensim = Word2Vec(words, vector_size=self.len_word_embed, window=5, min_count=1, workers=8)
        tag_embeddings_gensim = Word2Vec(tags, vector_size=self.len_tag_embed, window=5, min_count=1, workers=8)

        self.vocab = self.vocab.union(features_vocab)
        self.vocab = sorted(self.vocab)
        self.vocab2id = {self.vocab[i]: i for i in range(len(self.vocab))}

        self.pretrained_word_embeddings = torch.FloatTensor(len(self.word2id) + 1, self.len_word_embed)
        self.pretrained_tag_embeddings = torch.FloatTensor(len(self.tag2id) + 1, self.len_tag_embed)

        for w, id in self.word2id.items():
            self.pretrained_word_embeddings[id, :] = torch.tensor(word_embeddings_gensim.wv[w])

        for tag, id in self.tag2id.items():
            self.pretrained_tag_embeddings[id, :] = torch.tensor(tag_embeddings_gensim.wv[tag])

        self.word2id["unk"] = len(self.word2id)
        self.tag2id["unk"] = len(self.tag2id)

    def train(self):
        parser = NeuralParser(self.len_word_embed, self.len_tag_embed, self.word2id, self.tag2id, self.label2id,
                              self.vocab2id,
                              self.pretrained_word_embeddings,
                              self.pretrained_tag_embeddings).to(self.device)

        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(parser.parameters(), lr=0.002, betas=(0.9, 0.9), weight_decay=1e-6)

        f = list(read_conllu(filename=self.train_file))

        loss_training = []
        epochs = 109
        for epoch in range(epochs):
            loss_epoch = []
            for sent in f:
                opt.zero_grad()
                arcs_target = torch.tensor([int(sent[i].head) if i > 0 else 0 for i in range(len(sent))]).to(
                    self.device)
                labels_target = torch.tensor(
                    [self.label2id[sent[i].deprel] if i > 0 else self.label2id["root"] for i in range(len(sent))]).to(
                    self.device)

                adj_matrix, labels = parser.forward(sequence=sent)

                loss1 = loss_fn(adj_matrix, arcs_target)
                loss2 = loss_fn(labels, labels_target)

                loss = loss1 + loss2
                loss.backward()
                loss_epoch.append(loss.item())
                opt.step()

            mean_loss = sum(loss_epoch) / len(loss_epoch)
            loss_training.append(mean_loss)

            print(f"Step: {epoch + 1} Loss: {mean_loss}")

        model = {"State": parser.state_dict(), "OptState": opt.state_dict(), "Epochs": epochs,
                 "Loss": sum(loss_training) / len(loss_training), "len_word_embedding": self.len_word_embed,
                 "len_tag_embedding": self.len_tag_embed, "word2id": self.word2id, "tag2id": self.tag2id,
                 "label2id": self.label2id,
                 "vocab2id": self.vocab2id,
                 "pretrained_word": self.pretrained_word_embeddings, "pretrained_tag": self.pretrained_tag_embeddings,
                 "vocab": self.vocab}

        torch.save(model, "model/best_model.pt")

        print("model saved.")


if __name__ == '__main__':
    trainer = Trainer(train_file="UD_Bulgarian-BTB/bg_btb-ud-train.conllu")

    trainer.train()
