import tokenizer
import torch
import torch.nn as nn
#from ConvNet import Encoder, Decoder, Seq2Seq
from Transformer import Encoder, Decoder, EncoderLayer, DecoderLayer, SelfAttention, PositionwiseFeedforward, Seq2Seq
import RateOpt

from nltk.translate.bleu_score import sentence_bleu

import math
import time


PATH = 'data/wikismall/PWKP_108016.tag.80.aner.ori'
BATCH_SIZE = 128


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        #optimizer.zero_grad()
        optimizer.optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1])

        # output = [batch size, trg sent len - 1, output dim]
        # trg = [batch size, trg sent len]

        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg sent len - 1, output dim]
        # trg = [batch size * trg sent len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output, _ = model(src, trg[:, :-1])
            # output = [batch size, trg sent len - 1, output dim]
            # trg = [batch size, trg sent len]

            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg sent len - 1, output dim]
            # trg = [batch size * trg sent len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def train(model, train_iter, valid_iter, optimizer, criterion, nr_epochs=10, clip=1):
    print(f'The model has {count_parameters(model):,} trainable parameters')
    best_valid_loss = float('inf')

    for epoch in range(nr_epochs):

        start_time = time.time()

        train_loss = train_epoch(model, train_iter, optimizer, criterion, clip)
        valid_loss = evaluate(model, valid_iter, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model/conv_seq2seq.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iter, valid_iter, test_iter, src_field, dst_field = tokenizer.get_data(PATH, BATCH_SIZE, False)

    INPUT_DIM = len(src_field.vocab)
    OUTPUT_DIM = len(dst_field.vocab)
    EMB_DIM = 256
    HID_DIM = 512
    ENC_LAYERS = 10
    DEC_LAYERS = 10
    ENC_KERNEL_SIZE = 3
    DEC_KERNEL_SIZE = 3
    ENC_DROPOUT = 0.25
    DEC_DROPOUT = 0.25
    #PAD_IDX = dst_field.vocab.stoi['<pad>']
    PAD_IDX = src_field.vocab.stoi['<pad>']
    n_layers = 6
    n_heads = 8
    pf_dim = 2048
    dropout = 0.1

    #enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT, device)
    #dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, DEC_LAYERS, DEC_KERNEL_SIZE, DEC_DROPOUT, PAD_IDX, device)

    enc = Encoder(INPUT_DIM, HID_DIM, n_layers, n_heads, pf_dim, EncoderLayer, SelfAttention, PositionwiseFeedforward,
                  dropout, device)

    n_layers = 6
    n_heads = 8
    pf_dim = 2048
    dropout = 0.1

    dec = Decoder(OUTPUT_DIM, HID_DIM, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward,
                  dropout, device)

    model = Seq2Seq(enc, dec, PAD_IDX, device).to(device)

    #optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

    #use valid & test set to save time...
    optimizer = RateOpt.NoamOpt(HID_DIM, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    train(model, train_iter, valid_iter, optimizer, criterion)

    model.load_state_dict(torch.load('model/conv_seq2seq.pt'))
    test_loss=evaluate(model, test_iter, criterion)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

main()