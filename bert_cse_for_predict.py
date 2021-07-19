import paddle
import paddle.nn as nn
from bert_cse_common import tokenize, avg_pooling


class BertCseForPredict(nn.Layer):
    def __init__(self,model_path):
        super(BertCseForPredict, self).__init__()
        self.bert = paddle.jit.load(model_path)

    def forward(self, text):
        input_ids = tokenize([text])
        sequence_emb, cls_emb = self.bert(input_ids)
        text_emb = avg_pooling(sequence_emb)

        return  text_emb



