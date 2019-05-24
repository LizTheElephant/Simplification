import tokenizer
import torch
import Transformer
import ConvNet
import utils

import time
import tabulate
import copy
import csv
from zipfile import ZipFile
import datetime


# swa

# training
resume = False                             # resume training from checkpoint
save_freq = 20                              # frequency with which the model shall be saved
batch_size = 128
lr_init = 0.01                             # initial learning rate
momentum = 0.9
weight_decay = 1e-4

# data
cpt_directory = 'model'                    # checkpoint directory
csv_directory = 'csv'
date = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
columns = ['epoch', 'lr', 'swa_lr', 'cycle_length', 'tr_loss', 'val_loss', 'swa_loss', 'test_loss', 'swa_test_loss', 'time']
data_path = 'data/wikismall/PWKP_108016.tag.80.aner.ori'
zip_path = 'results.zip'

# network
n_heads = 8
pf_dim = 2048
transf_n_layers = 6
transf_dropout = 0.1
kernel_size = 3
conv_n_layers = 10
conv_dropout = 0.25


def train(model, train_iter, valid_iter, optimizer, criterion, pretrain_epochs, device, writer, cpt_filename):
    print(f'The model has {utils.count_parameters(model):,} trainable parameters')

    lr = lr_init

    utils.save_checkpoint(
        cpt_directory,
        1,
        date + "-" + cpt_filename,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict()
    )

    for epoch in range(pretrain_epochs):
        time_ep = time.time()
        swa = epoch > pretrain_epochs

        train_res = utils.train_epoch(model, train_iter, optimizer, criterion, device)
        valid_res = utils.evaluate(model, valid_iter, criterion, device)

        time_ep = time.time() - time_ep
        values = [epoch + 1, lr, None, None, train_res['loss'], valid_res['loss'], None, None, None, time_ep]
        writer.writerow(values)

        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
        if epoch % 20 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

        if (epoch + 1) % save_freq == 0 or swa:
            utils.save_checkpoint(
                cpt_directory,
                epoch + 1,
                date + "final-" + cpt_filename,
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict()
            )


def swa_train(model, swa_model, train_iter, valid_iter, optimizer, criterion, pretrain_epochs, swa_epochs, swa_lr, cycle_length, device, writer, cpt_filename):
    swa_n = 1

    swa_model.load_state_dict(copy.deepcopy(model.state_dict()))

    utils.save_checkpoint(
        cpt_directory,
        1,
        '{}-swa-{:2.4f}-{:03d}-{}'.format(date, swa_lr, cycle_length, cpt_filename),
        state_dict=model.state_dict(),
        swa_state_dict=swa_model.state_dict(),
        swa_n=swa_n,
        optimizer=optimizer.state_dict()
    )

    for epoch in range(pretrain_epochs + swa_epochs):
        epoch = epoch + pretrain_epochs
        time_ep = time.time()
        lr = utils.schedule(epoch, cycle_length, lr_init, swa_lr)
        utils.adjust_learning_rate(optimizer, lr)

        train_res = utils.train_epoch(model, train_iter, optimizer, criterion, device)
        valid_res = utils.evaluate(model, valid_iter, criterion, device)

        utils.moving_average(swa_model, model, swa_n)
        swa_n += 1
        utils.bn_update(train_iter, swa_model)
        swa_res = utils.evaluate(swa_model, valid_iter, criterion, device)

        time_ep = time.time() - time_ep
        values = [epoch + 1, lr, swa_lr, cycle_length, train_res['loss'], valid_res['loss'], swa_res['loss'], None, None, time_ep]
        writer.writerow(values)

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
            '{}-swa-{:2.4f}-{:03d}-{}'.format(date, swa_lr, cycle_length, cpt_filename),
            state_dict=model.state_dict(),
            swa_state_dict=swa_model.state_dict(),
            swa_n=swa_n,
            optimizer=optimizer.state_dict()
        )


def test(model, swa_model, test_iter, criterion, device, writer):
    time_ep = time.time()

    test_res = utils.evaluate(model, test_iter, criterion, device)
    swa_res = utils.evaluate(swa_model, test_iter, criterion, device)

    time_ep = time.time() - time_ep
    values = [None, None, None, None, None, None, None, test_res['loss'], swa_res['loss'], time_ep]
    writer.writerow(values)

    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    table = table.split('\n')[2]
    print(table)
    return [values]

def main(use_test_set, swa_epochs, pretrain_epochs):

    # setup
    utils.make_directory('model')
    utils.make_directory('csv')

    # experiment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iter, valid_iter, test_iter, src_field, dst_field = tokenizer.get_data(data_path, batch_size, False, use_test_set)

    input_dim = len(src_field.vocab)
    output_dim = len(dst_field.vocab)
    hidden_dim = 512
    embed_dem = 256
    pad_idx = src_field.vocab.stoi['<pad>']

    # TRANSFORMER
    enc = Transformer.Encoder(input_dim, hidden_dim, transf_n_layers, n_heads, pf_dim,
                              Transformer.EncoderLayer, Transformer.SelfAttention, Transformer.PositionwiseFeedforward,
                              transf_dropout, device)

    dec = Transformer.Decoder(output_dim, hidden_dim, transf_n_layers, n_heads, pf_dim,
                              Transformer.DecoderLayer, Transformer.SelfAttention, Transformer.PositionwiseFeedforward,
                              transf_dropout, device)

    model = Transformer.Seq2Seq(enc, dec, pad_idx, device).to(device)
    swa_model = Transformer.Seq2Seq(enc, dec, pad_idx, device).to(device)
    model_immutable = Transformer.Seq2Seq(enc, dec, pad_idx, device).to(device)

    criterion = torch.nn.functional.cross_entropy
    optimizer = torch.optim.SGD(model.parameters(), lr_init, momentum, weight_decay)

    cpt_filename = 'transformer'          # checkpoint filename baseline model

    swa_lr_set = [0.001, 0.05, 0.1]
    cycle_length_set = [5, 7, 10]

    print('TRANSFORMER')
    with open(utils.join_paths(csv_directory, cpt_filename +"-" + date + '.csv'), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)
        train(model, train_iter, valid_iter, optimizer, criterion, pretrain_epochs, device, writer, cpt_filename)
        model_immutable.load_state_dict(copy.deepcopy(model.state_dict()))
        for swa_lr in swa_lr_set:
            print('LEARNING RATE-{:2.4f}'.format(swa_lr))
            for cycle_length in cycle_length_set:
                print('CYCLE LENGTH-{:03d}'.format(cycle_length))
                model.load_state_dict(copy.deepcopy(model_immutable.state_dict()))
                swa_train(model, swa_model, train_iter, valid_iter, optimizer, criterion, pretrain_epochs, swa_epochs, swa_lr, cycle_length, device, writer, cpt_filename)
                test(model, swa_model, test_iter, criterion, device, writer)
    csv_file.close()

    # CONV NET
    enc = ConvNet.Encoder(input_dim, embed_dem, hidden_dim, conv_n_layers, kernel_size, conv_dropout, device)
    dec = ConvNet.Decoder(output_dim, embed_dem, hidden_dim, conv_n_layers, kernel_size, conv_dropout, pad_idx, device)

    model = ConvNet.Seq2Seq(enc, dec, device).to(device)
    swa_model = ConvNet.Seq2Seq(enc, dec, device).to(device)
    model_immutable = ConvNet.Seq2Seq(enc, dec, device).to(device)

    cpt_filename = 'convNet'  # checkpoint filename baseline model
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

    #print('CONV NET')
    #with open(utils.join_paths(csv_directory, cpt_filename + "-" + date + '.csv'), 'w', newline='') as csv_file:
    #    writer = csv.writer(csv_file)
    #    writer.writerow(columns)
    #    train(model, train_iter, valid_iter, optimizer, criterion, pretrain_epochs, device, writer, cpt_filename)
    #    model_immutable.load_state_dict(copy.deepcopy(model.state_dict()))
    #    for swa_lr in swa_lr_set:
    #        print('LEARNING RATE-{:2.4f}'.format(swa_lr))
    #        for cycle_length in cycle_length_set:
    #            print('CYCLE LENGTH-{:03d}'.format(cycle_length))
    #            model.load_state_dict(copy.deepcopy(model_immutable.state_dict()))
    #            swa_train(model, swa_model, train_iter, valid_iter, optimizer, criterion, pretrain_epochs, swa_epochs,
    #                      swa_lr, cycle_length, device, writer, cpt_filename)
    #            test(model, swa_model, test_iter, criterion, device, writer)
        #train(model, swa_model, train_iter, valid_iter, optimizer, criterion, pretrain_epochs, swa_epochs, device, writer, cpt_filename)
        #test(model, swa_model, test_iter, criterion, device, writer)
    #csv_file.close()

    # zip and upload
    file_paths = utils.get_all_file_paths(csv_directory) + utils.get_all_file_paths(cpt_directory)
    with ZipFile(zip_path, 'w') as zip:
        for file in file_paths:
            zip.write(file)

main(False, 20, 200)