import tokenizer
import torch
import transformer
import conv_net
import utils
import time
import tabulate
import copy
import csv
from zipfile import ZipFile
import datetime


# data
cpt_directory = 'model'
csv_directory = 'csv'
date = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'))
columns = ['epoch', 'lr', 'swa_lr', 'cycle_length', 'tr_loss', 'val_loss', 'swa_loss', 'test_loss', 'swa_test_loss', 'time']
data_path = 'data/wikismall/PWKP_108016.tag.80.aner.ori'
zip_path = 'results.zip'

# training 
resume = False
save_freq = 20
batch_size = 128
lr_init = 0.01
momentum = 0.9
weight_decay = 1e-4


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

    for e in range(swa_epochs):
        epoch = e + pretrain_epochs
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


def process_net(model, swa_model, model_immutable, train_iter, valid_iter, test_iter, 
                optimizer, criterion, swa_epochs, pretrain_epochs, cpt_filename, 
                swa_lr_set = [0.001, 0.05, 0.1], cycle_length_set = [1, 5, 10], device = 'cpu'):
    print('Training')
    with open(utils.join_paths(csv_directory, cpt_filename + "-" + date + '.csv'), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)
        train(model, train_iter, valid_iter, optimizer, criterion, pretrain_epochs, device, writer, cpt_filename)
        model_immutable.load_state_dict(copy.deepcopy(model.state_dict()))
    csv_file.close()

    print('SWA Training')

    for swa_lr in swa_lr_set:
        print('LEARNING RATE-{:2.4f}'.format(swa_lr))
        for cycle_length in cycle_length_set:
            with open(utils.join_paths(csv_directory, '{}-SWA-{:03}-{}-{}.csv'.format(cpt_filename, swa_lr, cycle_length, date)), 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(columns)
                print('CYCLE LENGTH-{:03}'.format(cycle_length))
                model.load_state_dict(copy.deepcopy(model_immutable.state_dict()))
                swa_train(model, swa_model, train_iter, valid_iter, optimizer, criterion, pretrain_epochs, swa_epochs,
                          swa_lr, cycle_length, device, writer, cpt_filename)
                test(model, swa_model, test_iter, criterion, device, writer)


def main(swa_epochs, pretrain_epochs):

    utils.make_directory('model')
    utils.make_directory('csv')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iter, valid_iter, test_iter, src_field, dst_field = tokenizer.get_data(data_path, batch_size, False)

    INPUT_DIM = len(src_field.vocab)
    OUTPUT_DIM = len(dst_field.vocab)
    HID_DIM = 512
    EMB_DIM = 256
    PAD_IDX = src_field.vocab.stoi['<pad>']

    # TRANSFORMER
    enc = transformer.Encoder(INPUT_DIM, HID_DIM, transf_n_layers, n_heads, pf_dim,
                  transformer.EncoderLayer, transformer.SelfAttention, transformer.PositionwiseFeedforward,
                  transf_dropout, device)

    dec = transformer.Decoder(OUTPUT_DIM, HID_DIM, transf_n_layers, n_heads, pf_dim,
                  transformer.DecoderLayer, transformer.SelfAttention, transformer.PositionwiseFeedforward,
                  transf_dropout, device)

    model = transformer.Seq2Seq(enc, dec, PAD_IDX, device).to(device)
    swa_model = transformer.Seq2Seq(enc, dec, PAD_IDX, device).to(device)
    model_immutable = transformer.Seq2Seq(enc, dec, PAD_IDX, device).to(device)

    process_net(model, swa_model, model_immutable, 
                train_iter, valid_iter, test_iter,
                optimizer = torch.optim.SGD(model.parameters(), 
                criterion = torch.nn.functional.cross_entropy,
                swa_epochs = swa_epochs, pretrain_epochs = pretrain_epochs, 
                cpt_filename = 'transformer', 
                device = device):

    # CONV NET
    enc = conv_net.Encoder(INPUT_DIM, EMB_DIM, HID_DIM, conv_n_layers, kernel_size, conv_dropout, device)
    dec = conv_net.Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, conv_n_layers, kernel_size, conv_dropout, PAD_IDX, device)

    model = conv_net.Seq2Seq(enc, dec, device).to(device)
    swa_model = conv_net.Seq2Seq(enc, dec, device).to(device)
    model_immutable = conv_net.Seq2Seq(enc, dec, device).to(device)

    process_net(model, swa_model, model_immutable, 
                train_iter, valid_iter, test_iter,
                optimizer = torch.optim.Adam(model.parameters()),
                criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX),
                swa_epochs = swa_epochs, pretrain_epochs = pretrain_epochs, 
                cpt_filename = 'conv_net', 
                device = device):

    # zip and upload
    file_paths = utils.get_all_file_paths(csv_directory) + utils.get_all_file_paths(cpt_directory)
    with ZipFile(zip_path, 'w') as zip:
        for file in file_paths:
            zip.write(file)


main(15, 50)
