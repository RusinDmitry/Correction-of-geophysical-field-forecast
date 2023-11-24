import sys

sys.path.insert(0, '../')
import torch
from correction.config import cfg
from tqdm import tqdm
import os


def train(train_dataloader, valid_dataloader, model, optimizer, wrf_scaler, era_scaler,
          criterion, lr_scheduler, logger, max_epochs):
    for epoch in range(max_epochs):
        train_loss = train_epoch(train_dataloader, model, criterion,
                                 optimizer, wrf_scaler, era_scaler)
        if logger:
            logger.train_loss.append(train_loss)
        print('train loss', train_loss)
        valid_loss = eval_epoch(model, criterion, wrf_scaler, era_scaler, valid_dataloader, logger, epoch)
        print('valid_loss', valid_loss)
        lr_scheduler.step()
        print(lr_scheduler.get_last_lr())
        if logger:
            best_epoch = logger.save_model(model.state_dict(), epoch)
        else:
            torch.save(model.state_dict(), os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, f'model_{epoch}.pth'))


def train_epoch(dataloader, model, criterion, optimizer, wrf_scaler, era_scaler):
    train_loss = 0
    model.train()

    for train_data, train_label in (pbar := tqdm(dataloader)):
        train_data = torch.swapaxes(train_data.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
        train_label = torch.swapaxes(train_label.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
        train_data = wrf_scaler.channel_transform(train_data, 2)
        train_label = era_scaler.channel_transform(train_label, 2)

        optimizer.zero_grad()

        output = model(train_data)

        loss = criterion(train_data, output, train_label)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=50.0)
        optimizer.step()

        l = loss.detach().item()
        train_loss += l
        pbar.set_description(f'{l}')

    return train_loss / len(dataloader)


def eval_epoch(model, criterion, wrf_scaler, era_scaler, dataloader, logger, epoch=None):
    with torch.no_grad():
        model.eval()
        valid_loss = 0.0
        for valid_data, valid_label in tqdm(dataloader):
            valid_data = torch.swapaxes(valid_data.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
            valid_label = torch.swapaxes(valid_label.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
            valid_data = wrf_scaler.channel_transform(valid_data, 2)
            valid_label = era_scaler.channel_transform(valid_label, 2)

            output = model(valid_data)

            loss = criterion(valid_data, output, valid_label, logger)
            valid_loss += loss.item()
        valid_loss = valid_loss / len(dataloader)
        if logger:
            logger.print_stat_readable(epoch)
    return valid_loss
