import re
class Token:

    def __init__(self, form=None, lemma=None, upos=None, feat=None,
                 head=None, deprel=None):
        self.form = form
        self.lemma = lemma
        self.upos = upos
        self.feat = feat
        self.head = None if head is None or head == '_' else head
        self.deprel = deprel

    def copy(self):
        return Token(form=self.form, lemma=self.lemma, upos=self.upos,
                     feat=self.feat, head=self.head, deprel=self.deprel)


def read_conllu(filename):
    with open(filename, encoding="utf8") as f:
        sentence = [Token('<ROOT>')]
        for line in f:
            if len(line.strip()) == 0:  # sentence boundary
                yield sentence
                sentence = [Token('<ROOT>')]
                continue
            if line.startswith('#'):  # comments
                continue

            index, form, lemma, upos, _, feats, \
                head, deprel, _, _ = line.split('\t')
            if not re.match('^[0-9]+$', index):  # not a basic dep. line
                continue
            sentence.append(Token(form=form, lemma=lemma, upos=upos,
                                  feat=feats, head=head, deprel=deprel))
