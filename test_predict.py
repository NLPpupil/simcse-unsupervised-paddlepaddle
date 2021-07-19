import paddle.nn as nn

from bert_cse_common import cos_sim
from bert_cse_for_predict import BertCseForPredict

def test():
    bert_cse = BertCseForPredict('/root/paddlejob/workspace/output/simcse_model/bert_cse')
    a1 = '我是你爸爸'
    a2 = '我是你父亲'
    b1 = '今天天气不错'
    a1_emb = bert_cse(a1)
    a2_emb = bert_cse(a2)
    b1_emb = bert_cse(b1)

    sim_a1a2 = cos_sim(a1_emb, a2_emb)
    sim_a1b1 = cos_sim(a1_emb, b1_emb)

    print (sim_a1a2)
    print (sim_a1b1)

if __name__ == '__main__':
    test()




