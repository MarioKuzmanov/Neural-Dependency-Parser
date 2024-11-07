import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from model import NeuralParser


class NeuralScorer:

    def __init__(self):
        self.parser_loaded, self.deprel2id, self.id2deprel = None, None, None

    def load_model(self):
        saved = torch.load("model/best_model.pt", map_location="cpu")

        print("Train Steps: {}".format(saved["Epochs"]))

        self.parser_loaded = NeuralParser(saved["len_word_embedding"], saved["len_tag_embedding"], saved["word2id"],
                                          saved["tag2id"],
                                          saved["label2id"],
                                          saved["vocab2id"],
                                          saved["pretrained_word"],
                                          saved["pretrained_tag"]).cpu()

        self.parser_loaded.load_state_dict(saved["State"])
        self.deprel2id = saved["label2id"]
        self.id2deprel = {v: k for k, v in saved["label2id"].items()}

    def scorer(self, sent):
        adjacency_matrix, labels = self.parser_loaded.forward(sent)

        return adjacency_matrix, labels
