import paddle
import paddle.nn as nn

from file_utils import read_data
from bert_cse_for_train import BertCseForTrain
from paddle.io import DataLoader
from dataset import SimCSEUnsupervisedDataset
from paddle.static import InputSpec
from bert_cse_common import config


def train():
    model = BertCseForTrain()
    model.train()
    epochs = int(config['num_epochs'])
    batch_size = int(config['batch_size'])
    optim = paddle.optimizer.Adam(learning_rate=0.00001, parameters=model.parameters())
    train_dataset = SimCSEUnsupervisedDataset(read_data(config['data_path']))
    cross_entropy = nn.loss.CrossEntropyLoss(use_softmax=True)
    labels = paddle.arange(batch_size)

    step = 0
    verbose_every_n_steps = 10
    for i in range(epochs):
        train_dataset.shuffle()
        print('epoch {0}/{1}'.format(i + 1, epochs))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
        for texts, dummy_label in train_loader():
            sim = model(texts)
            loss = cross_entropy(sim, labels)
            loss.backward()
            optim.step()
            optim.clear_grad()
            if step % verbose_every_n_steps == 0:
                print('{0} steps loss={1}'.format(step, loss.numpy()[0]))
            step += 1

    path = "/root/paddlejob/workspace/output/simcse_model/bert_cse"
    paddle.jit.save(
        layer=model.bert,
        path=path,
        input_spec=[InputSpec(shape=[None, 128], dtype='int64')])



if __name__ == '__main__':
    train()
