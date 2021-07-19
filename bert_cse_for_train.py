import paddle
from paddlenlp.transformers import BertModel
from numpy import repeat
from bert_cse_common import *


class BertCseForTrain(nn.Layer):
    def __init__(self):
        super(BertCseForTrain, self).__init__()
        self.bert = BertModel.from_pretrained(config['bert_name'])
        self.temperature = float(config['temperature'])

    def forward(self, texts):
        texts = repeat(texts, 2).tolist()
        input_ids = tokenize(texts)
        sequence_emb, cls_emb = self.bert(input_ids)
        text_emb = avg_pooling(sequence_emb)
        text_emb = paddle.reshape(text_emb, [-1, 2, 768])
        text_emb_original = text_emb[:, 0]
        text_emb_copy = text_emb[:, 1]
        sim = cos_sim(text_emb_original.unsqueeze(1), text_emb_copy.unsqueeze(0)) / self.temperature

        return sim

