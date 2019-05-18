import tokenizer
import torch
import torch.nn as nn
from Transformer import Encoder, Decoder, EncoderLayer, DecoderLayer, SelfAttention, PositionwiseFeedforward, Seq2Seq
import utils

import time
import tabulate


swa_start = 20                         # number of epoch after which SWA will start to average models
swa_lr = 0.05                          # swa learning rate
swa_c_epochs = 1                       # SWA model collection frequency / cycle length (epochs)
eval_freq = 3                          # frequency with which the model shall be evaluated
save_freq = 6                          # frequency with which the model shall be saved
resume = False                         # resume training from checkpoint
cpt_directory = 'model'                # checkpoint directory
cpt_filename = 'transformer_swa.pt'    # checkpoint filename

data_path = 'data/wikismall/PWKP_108016.tag.80.aner.ori'
batch_size = 128
nr_epochs=15
lr_init = 0.01                         # initial learning rate
momentum = 0.9
weight_decay = 1e-4
n_layers = 6
n_heads = 8
pf_dim = 2048
dropout = 0.1


def train(model, swa_model, train_iter, valid_iter, optimizer, criterion, swa_n = 0):
    print(f'The model has {utils.count_parameters(model):,} trainable parameters')
    swa_res = {'loss': None}
    start_epoch = 0
    columns = ['ep', 'lr', 'tr_loss', 'val_loss', 'swa_val_loss', 'time']

    if resume:
        print('Resume training from %s' % cpt_filename)
        checkpoint = torch.load(cpt_filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        swa_state_dict = checkpoint['swa_state_dict']
        if swa_state_dict is not None:
            swa_model.load_state_dict(swa_state_dict)
        swa_n_ckpt = checkpoint['swa_n']
        if swa_n_ckpt is not None:
            swa_n = swa_n_ckpt

    utils.save_checkpoint(
        cpt_directory,
        start_epoch,
        cpt_filename,
        state_dict=model.state_dict(),
        swa_state_dict=swa_model.state_dict(),
        swa_n=swa_n,
        optimizer=optimizer.state_dict()
    )

    for epoch in range(nr_epochs):
        time_ep = time.time()

        lr = utils.schedule(epoch, swa_start, swa_lr, lr_init)
        utils.adjust_learning_rate(optimizer, lr)
        train_res = utils.train_epoch(model, train_iter, optimizer, criterion)

        if epoch == 0 or epoch % eval_freq == eval_freq - 1 or epoch == nr_epochs - 1:
            valid_res = utils.evaluate(model, valid_iter, criterion)
        else:
            valid_res = {'loss': None}

        if (epoch + 1) >= swa_start and (epoch + 1 - swa_start) % swa_c_epochs == 0:
            utils.moving_average(swa_model, model, 1.0 / (swa_n + 1))
            swa_n += 1
            if epoch == 0 or epoch % eval_freq == eval_freq - 1 or epoch == nr_epochs - 1:
                utils.bn_update(train_iter, swa_model)
                swa_res = utils.eval(swa_model, valid_iter, criterion)
            else:
                swa_res = {'loss': None, 'accuracy': None}

        if (epoch + 1) % save_freq == 0:
            utils.save_checkpoint(
                cpt_directory,
                epoch + 1,
                cpt_filename,
                state_dict=model.state_dict(),
                swa_state_dict=swa_model.state_dict(),
                swa_n=swa_n,
                optimizer=optimizer.state_dict()
            )

        time_ep = time.time() - time_ep
        values = [epoch + 1, lr, train_res['loss'],valid_res['loss'], swa_res['loss'], time_ep]

        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
        if epoch % 10 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

    if (epoch + 1) % save_freq == 0:
        utils.save_checkpoint(
            cpt_directory,
            epoch + 1,
            cpt_filename,
            state_dict=model.state_dict(),
            swa_state_dict=swa_model.state_dict(),
            swa_n=swa_n,
            optimizer=optimizer.state_dict()
        )

    return swa_n

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
    swa_model = Seq2Seq(enc, dec, pad_idx, device).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)
    optimizer = torch.optim.SGD(model.parameters(), lr_init, momentum, weight_decay)

    train(model, swa_model, train_iter, valid_iter, optimizer, criterion)

    test_res = utils.evaluate(model, test_iter, criterion)
    test_loss = test_res['loss']
    print(f'| Test Loss: {test_loss}')

main()