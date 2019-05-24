import torchtext.data
from torchtext.datasets import TranslationDataset
from torchtext.data import Field, BucketIterator
from spacy.lang.en import English
import dill

def tokenize(text):
    return [tok.text for tok in English().tokenizer(text)]


def untokenize(doc, index, vocab):
    eos_tok = vocab.stoi['<eos>']
    length = (doc[index] == eos_tok).nonzero()[0]
    return ' '.join([vocab.itos[tok] for tok in doc[index][1:length]])


def save_data(path, field):
    with open(path, 'wb') as f:
        dill.dump(field, f)


def load_data(path):
    with open(path, 'rb') as f:
        return dill.load(f)


def initialize_field(path_src, path_dst, load):
    if load:
        return load_data(path_src), load_data(path_dst)

    else:
        return Field(tokenize=tokenize, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True),\
               Field(tokenize=tokenize, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)


def get_data(path, batch_size, load, mini=True):

    field_src, field_dst = initialize_field(path + 'field.src', path + 'field.dst', load)

    # Load data
    print("Loading Training Set... ")
    #train_set = TranslationDataset(path=path + '.train.', exts=('src', 'dst'), fields=(field_src, field_dst))

    train_set = TranslationDataset(path=path + ('.test.' if mini else '.train.'), exts=('src', 'dst'), fields=(field_src, field_dst))

    print("Loading Validation Set... ")
    valid_set = TranslationDataset(path=path + ('.valid.' if mini else '.valid.'), exts=('src', 'dst'), fields=(field_src, field_dst))

    print("Loading Test Set... ")
    test_set  = TranslationDataset(path=path + ('.test.' if mini else '.test.'), exts=('src', 'dst'), fields=(field_src, field_dst))

    # Build vocabulary. Train, validation and test sets share the same volcabulary
    if load==False:
        print("Build vocabulary... ")
        field_src.build_vocab(valid_set)
        field_dst.build_vocab(valid_set)
        save_data(path + '.field.src', field_src)
        save_data(path + '.field.dst', field_dst)

    # Initialize dataloaders
    print("Creating Iterators... ")
    train_iter = BucketIterator(dataset=train_set, batch_size=batch_size,
                                sort_key=lambda x: torchtext.data.interleave_keys(len(x.field_src), len(x.field_dst)))
    valid_iter = BucketIterator(dataset=valid_set, batch_size=batch_size,
                                sort_key=lambda x: torchtext.data.interleave_keys(len(x.field_src), len(x.field_dst)))
    test_iter  = BucketIterator(dataset=test_set, batch_size=batch_size,
                                sort_key=lambda x: torchtext.data.interleave_keys(len(x.field_src), len(x.field_dst)))
    return train_iter, valid_iter, test_iter, field_src, field_dst
    #return train_set, test_iter, test_iter, field_src, field_dst