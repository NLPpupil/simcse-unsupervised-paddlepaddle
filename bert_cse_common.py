from paddle import nn
from paddle import to_tensor
from paddlenlp.transformers import BertTokenizer
from file_utils import read_config

config = read_config()
cos_sim = nn.CosineSimilarity(axis=-1)
tokenizer = BertTokenizer.from_pretrained(config['bert_name'])
max_seq_len = int(config['max_seq_len'])


def tokenize(texts):
    tokenize_results = tokenizer(texts, max_seq_len=max_seq_len, pad_to_max_seq_len=True, return_token_type_ids=False)
    input_ids = [r['input_ids'] for r in tokenize_results]
    return to_tensor(input_ids)


def avg_pooling(sequence_emb):
    text_emb = sequence_emb.sum(1) / max_seq_len
    return text_emb
