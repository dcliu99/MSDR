import os
import numpy as np
import math
import json
import torch
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm
import copy, time
from utils.util import save_model, get_number_of_parameters
from collections import defaultdict
from utils.evaluate import evaluate
from utils.util import MyEncoder


def train_model(model: nn.Module,
                   dataloaders,
                   optimizer,
                   normal,
                   scheduler,
                   folder: str,
                   trainer,
                   tensorboard_folder,
                   epochs: int,
                   device,
                   max_grad_norm: float = None,
                   early_stop_steps: float = None):

    save_path = os.path.join(folder, 'best_model.pkl')

    if os.path.exists(save_path):
        print("path exist")
        save_dict = torch.load(save_path)
        model.load_state_dict(save_dict['model_state_dict'])
        optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        best_val_loss = save_dict['best_val_loss']
        begin_epoch = save_dict['epoch'] + 1
    else:
        print("path does not exist")
        save_dict = dict()
        best_val_loss = float('inf')
        begin_epoch = 0

    phases = ['train', 'validate', 'test']
    writer = SummaryWriter(tensorboard_folder)
    since = time.perf_counter()
    model = model.to(device)
    print(f'Trainable parameters: {get_number_of_parameters(model)}.')

    try:
        for epoch in range(begin_epoch, begin_epoch + epochs):
            running_loss, running_metrics = defaultdict(float), dict()
            for phase in phases:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                steps, predictions, running_targets = 0, list(), list()
                tqdm_loader = tqdm(enumerate(dataloaders[phase]))
                for step, (inputs, targets) in tqdm_loader:
                    running_targets.append(targets.numpy())

                    with torch.no_grad():
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, loss = trainer.train(inputs, targets, phase)

                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            if max_grad_norm is not None:

                                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            optimizer.step()

                    with torch.no_grad():
                        predictions.append(outputs.cpu().numpy())
                    running_loss[phase] += loss.item() * len(targets)
                    steps += len(targets)

                    tqdm_loader.set_description(
                        f'{phase:5} epoch: {epoch:3}, {phase:5} loss: {normal[0].rmse_transform(running_loss[phase] / steps):3.6}')
                    torch.cuda.empty_cache()
                running_metrics[phase] = evaluate(np.concatenate(predictions), np.concatenate(running_targets), normal)
                running_metrics[phase].pop('rmse')
                running_metrics[phase].pop('pcc')
                running_metrics[phase].pop('mae')
                if phase == 'validate':
                    model.eval()
                    if running_loss['validate'] <= best_val_loss or math.isnan(running_loss['validate']):
                        best_val_loss = running_loss['validate']
                        save_dict.update(model_state_dict=copy.deepcopy(model.state_dict()),
                                         epoch=epoch,
                                         best_val_loss=best_val_loss,
                                         optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))
                        save_model(save_path, **save_dict)
                        print(f'Better model at epoch {epoch} recorded.')
                        process_test(folder, trainer, model, normal, dataloaders, device, epoch)
                    elif epoch - save_dict['epoch'] > early_stop_steps:
                        raise ValueError('Early stopped.')
            scheduler.step(running_loss['train'])

            for metric in running_metrics['train'].keys():
                for phase in phases:
                    for key, val in running_metrics[phase][metric].items():
                        writer.add_scalars(f'{metric}/{key}', {f'{phase}': val}, global_step=epoch)
            writer.add_scalars('Loss', {
                f'{phase} loss': running_loss[phase] / len(dataloaders[phase].dataset) for phase in phases},
                               global_step=epoch)
    except (ValueError, KeyboardInterrupt):
        writer.close()
        time_elapsed = time.perf_counter() - since
        print(f"cost {time_elapsed} seconds")
        print(f'model of epoch {save_dict["epoch"]} successfully saved at `{save_path}`')

def process_test(folder: str,
            trainer,
            model,
            normal,
            dataloaders,
            device,
            epoch):
    save_path = os.path.join(folder, 'best_model.pkl')
    save_dict = torch.load(save_path)
    model.load_state_dict(save_dict['model_state_dict'])
    model.eval()
    steps, predictions, running_targets = 0, list(), list()
    tqdm_loader = tqdm(enumerate(dataloaders['test']))
    for step, (inputs, targets) in tqdm_loader:
        running_targets.append(targets.numpy())

        with torch.no_grad():
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs, loss = trainer.train(inputs, targets, 'test')
            predictions.append(outputs.cpu().numpy())

    running_targets, predictions = np.concatenate(running_targets, axis=0), np.concatenate(predictions, axis=0)
    scores = evaluate(running_targets, predictions,normal)
    print('test results in epoch '+str(epoch)+':')
    print("\t rmse: "+json.dumps(scores['rmse'])+",")
    print("\t mae: "+json.dumps(scores['mae'])+",")
    print("\t pcc: "+json.dumps(scores['pcc'])+",")

def test_model(folder: str,
                  trainer,
                  model,
                  normal,
                  dataloaders,
                  conf,
                  device):

    save_path = os.path.join(folder, 'best_model.pkl')
    save_dict = torch.load(save_path)
    model.load_state_dict(save_dict['model_state_dict'])
    model.eval()
    steps, predictions, running_targets = 0, list(), list()
    tqdm_loader = tqdm(enumerate(dataloaders['test']))
    for step, (inputs, targets) in tqdm_loader:
        running_targets.append(targets.numpy())

        with torch.no_grad():
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs, loss = trainer.train(inputs, targets, 'test')
            predictions.append(outputs.cpu().numpy())

    running_targets, predictions = np.concatenate(running_targets, axis=0), np.concatenate(predictions, axis=0)

    scores = evaluate(running_targets, predictions,normal)
    print('test results:')
    print(json.dumps(scores,cls=MyEncoder, indent=4))

    if trainer.model.graph0 is not None:
        np.save(os.path.join(folder, 'graph0'),trainer.model.graph0.detach().cpu().numpy())
        np.save(os.path.join(folder, 'graph1'),trainer.model.graph1.detach().cpu().numpy())
        np.save(os.path.join(folder, 'graph2'),trainer.model.graph2.detach().cpu().numpy())

    np.savez(os.path.join(folder, 'test-results.npz'), predictions=predictions, targets=running_targets)