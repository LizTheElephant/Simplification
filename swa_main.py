import tokenizer
import torch
import torch.nn.functional as F
from Transformer import Encoder, Decoder, EncoderLayer, DecoderLayer, SelfAttention, PositionwiseFeedforward, Seq2Seq
import utils

import time
import tabulate
import copy
import csv
import datetime

pretrain_epochs = 50                       # number of epoch after which SWA will start to average models
swa_epochs = 20                            # number of epoch to average after training
swa_lr = 0.05                              # swa learning rate
eval_freq = 5                              # frequency with which the model shall be evaluated
save_freq = 25                             # frequency with which the model shall be saved
resume = False                             # resume training from checkpoint
cpt_directory = 'model'                    # checkpoint directory
cpt_filename = 'transformer'               # checkpoint filename baseline model
cpt_swa_filename = 'transformer_swa'       # checkpoint filename swa model
csv_directory = 'csv'
save_freq = 20

data_path = 'wikismall/PWKP_108016.tag.80.aner.ori'
batch_size = 128

cycle_length = 5
lr_init = 0.01                             # initial learning rate
momentum = 0.9
weight_decay = 1e-4
n_layers = 6
n_heads = 8
pf_dim = 2048
dropout = 0.1


def schedule(epoch):
    t = (((epoch-1) % cycle_length)+1)/cycle_length
    lr = (1-t)*swa_lr + t*lr_init
    return lr


def train(model, train_iter, valid_iter, optimizer, criterion, columns, csv_data, device):
    print(f'The model has {utils.count_parameters(model):,} trainable parameters')

    utils.save_checkpoint(
        cpt_directory,
        1,
        cpt_filename + str(datetime.datetime.now()).split('.')[0],
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict()
    )

    for epoch in range(pretrain_epochs):
        time_ep = time.time()

        train_res = utils.train_epoch(model, train_iter, optimizer, criterion, device)
        valid_res = utils.evaluate(model, valid_iter, criterion, device)

        time_ep = time.time() - time_ep
        values = [epoch + 1, train_res['loss'], valid_res['loss'], None, time_ep]

        csv_data.append(values)

        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
        if epoch % 20 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

        if (epoch + 1) % save_freq == 0:
            utils.save_checkpoint(
                cpt_directory,
                epoch + 1,
                cpt_filename + str(datetime.datetime.now()).split('.')[0],
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict()
            )

    return csv_data


def train_with_swa(model, swa_model, train_iter, valid_iter, optimizer, criterion, columns, csv_data, device):
    print(f'The model has {utils.count_parameters(model):,} trainable parameters')

    swa_n = 1

    utils.save_checkpoint(
        cpt_directory,
        1,
        cpt_swa_filename + str(datetime.datetime.now()).split('.')[0],
        state_dict=model.state_dict(),
        swa_state_dict=swa_model.state_dict(),
        swa_n=swa_n,
        optimizer=optimizer.state_dict()
    )

    for epoch in range(swa_epochs):
        time_ep = time.time()

        lr = schedule(epoch)
        utils.adjust_learning_rate(optimizer, lr)

        train_res = utils.train_epoch(model, train_iter, optimizer, criterion, device)
        valid_res = utils.evaluate(model, valid_iter, criterion, device)

        utils.moving_average(swa_model, model, swa_n)
        swa_n += 1
        utils.bn_update(train_iter, swa_model)
        swa_res = utils.evaluate(swa_model, valid_iter, criterion, device)

        time_ep = time.time() - time_ep
        values = [epoch + 1, lr, train_res['loss'], valid_res['loss'], swa_res['loss'], None, None, time_ep]
        csv_data.append(values)

        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
        if epoch % 20 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

        utils.save_checkpoint(
            cpt_directory,
            epoch + 1,
            cpt_swa_filename + str(datetime.datetime.now()).split('.')[0],
            state_dict=model.state_dict(),
            swa_state_dict=swa_model.state_dict(),
            swa_n=swa_n,
            optimizer=optimizer.state_dict()
        )

    return csv_data


def test(model, swa_model, test_iter, criterion, columns, csv_data, device):
    time_ep = time.time()

    test_res = utils.evaluate(model, test_iter, criterion, device)
    if swa_model is not None:
        swa_res = utils.evaluate(swa_model, test_iter, criterion, device)
    else:
        swa_res = {'loss': None}

    time_ep = time.time() - time_ep
    values = [None, None, None, None, None, test_res['loss'], swa_res['loss'], time_ep]

    csv_data.append(values)

    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    table = table.split('\n')[2]
    print(table)
    return csv_data


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iter, valid_iter, test_iter, src_field, dst_field = tokenizer.get_data(data_path, batch_size, False)

    input_dim = len(src_field.vocab)
    output_dim = len(dst_field.vocab)
    hidden_dim = 512
    pad_idx = src_field.vocab.stoi['<pad>']

    enc = Encoder(input_dim, hidden_dim, n_layers, n_heads, pf_dim,
                  EncoderLayer, SelfAttention, PositionwiseFeedforward,
                  dropout, device)

    dec = Decoder(output_dim, hidden_dim, n_layers, n_heads, pf_dim,
                  DecoderLayer, SelfAttention, PositionwiseFeedforward,
                  dropout, device)

    model = Seq2Seq(enc, dec, pad_idx, device).to(device)

    criterion = F.cross_entropy
    optimizer = torch.optim.SGD(model.parameters(), lr_init, momentum, weight_decay)

    columns = ['epoch', 'lr', 'tr_loss', 'val_loss', 'swa_loss', 'test_loss', 'swa_test_loss', 'time']
    csv_data = []
    csv_data.append(columns)
    csv_data = train(model, train_iter, valid_iter, optimizer, criterion, columns, csv_data, device)
    csv_data = test(model, test_iter, criterion, columns, csv_data, device)

    model_swa = Seq2Seq(enc, dec, pad_idx, device).to(device)
    model_swa.load_state_dict(copy.deepcopy(model.state_dict()))
    csv_data = train_with_swa(model, None, train_iter, valid_iter, optimizer, columns, csv_data, criterion)

    with open(csv_directory + '/total-' + str(datetime.datetime.now()).split('.')[0] + '.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(csv_data)

    csv_file.close()

main()