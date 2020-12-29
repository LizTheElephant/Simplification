import torch
import os

def schedule(epoch, cycle_length, min_lr, max_lr):
    t = (((epoch-1) % cycle_length)+1)/cycle_length
    lr = (1-t)*max_lr + t*min_lr
    return lr


def save_checkpoint(dir, epoch, filecode, **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    torch.save(state, join_paths(dir, (filecode + '-%d.pt') % epoch))


def join_paths(path1, path2):
    return os.path.join(path1, path2)


def make_directory(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_all_file_paths(directory):
    file_paths = []

    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def moving_average(swa_model, model, swa_n):
    for swa_param, model_param in zip(swa_model.parameters(), model.parameters()):
        swa_param.data = (swa_n * swa_param.data  + swa_param.data) / (swa_n + 1)


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, iterator, optimizer, criterion, device):
    model.train()

    epoch_loss = 0.0
    correct = 0.0

    for i, batch in enumerate(iterator):
        src = batch.src.to(device)
        trg = batch.trg.to(device)
        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1])
        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        prediction = output.data.max(1, keepdim=True)[1]
        correct += prediction.eq(trg).sum().item()

    return {
        'loss': epoch_loss / len(iterator),
        'accuracy': correct / len(iterator) * 1
    }


def evaluate(model, iterator, criterion, device):
    model.eval()

    epoch_loss = 0.0
    correct = 0.0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src.to(device)
            trg = batch.trg.to(device)

            output, _ = model(src, trg[:, :-1])
            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()
            prediction = output.data.max(1, keepdim=True)[1]
            correct += prediction.eq(trg).sum().item()
    return {
        'loss': epoch_loss / len(iterator),
        'accuracy': correct / len(iterator) * 100
    }
