import os
import shutil
import torch
import copy
import time
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# from utils.metric import calculate_metrics
from data_loader.data_container import load_dataset
from model.graphconv_model import GraphWavenet
from model.dcrnn_model import DCRNNmodel
from utils.util import create_kernel, convert_to_gpu
from utils.loss_new import create_loss, calculate_scores
from utils.result_writer import ResultWriter


def train_main(args):

    writer = ResultWriter(args['log_on'])

    print(f"train  {args['model_name']}  {args['save_name']}")

    data_loaders, scaler = load_dataset(args)

    supports = create_kernel(args)

    model = convert_to_gpu(eval(args['model_name'])(supports, args))

    # model = convert_to_gpu(DCRNNmodel(supports, args))

    loss_funcs = create_loss()

    optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])

    save_dict, worst_rmse = {'model_state_dict': copy.deepcopy(model.state_dict()), 'epoch': 0}, torch.tensor(100000)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.2, patience=5, threshold=1e-3, min_lr=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, args['lr_decay'])

    kwargs = {'scaler': scaler, 'global_step': 0}
    phases = ['train', 'val', 'test']

    for epoch in range(args['num_epochs']):
        running_loss = {phase: 0.0 for phase in phases}

        for phase in phases:
            if phase == 'train':
                model.train()
                kwargs['is_eval'] = False
            else:
                model.eval()
                kwargs['is_eval'] = True

            steps = 0
            predictions_torch, targets_torch = [], []
            tqdm_loader = tqdm(enumerate(data_loaders[phase]))
            ite = 0
            for step, (features, truth_data) in tqdm_loader:
                ite = ite + 1
                if ite > 10:
                    break
                features = convert_to_gpu(features)
                truth_data = convert_to_gpu(truth_data)
                kwargs['global_step'] += (1 if phase == 'train' else 0)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(features, **kwargs)
                    outputs = outputs.squeeze()
                    loss = loss_funcs[args['loss_function']](truth=truth_data, predict=outputs, scaler=scaler)
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        writer.add_scalar("Loss-"+phase+"-RMSE", loss, kwargs['global_step'])

                with torch.no_grad():
                    targets_torch.append(truth_data)
                    predictions_torch.append(outputs)
                    running_loss[phase] += loss * truth_data.size(0)
                    steps += truth_data.size(0)
                    tqdm_loader.set_description(f'{phase} epoch: {epoch}, {phase} loss: {running_loss[phase] / steps}')

                torch.cuda.empty_cache()

            with torch.no_grad():
                predictions_torch = torch.cat(predictions_torch, dim=0)
                targets_torch = torch.cat(targets_torch, dim=0)
                scores = calculate_scores(loss_funcs, predictions_torch, targets_torch, scaler)
                writer.add_scalar(f"{phase}/RMSE", scores['RMSE'], epoch)
                writer.add_scalar(f"{phase}/MAE", scores['MAE'], epoch)
                writer.add_scalar(f"{phase}/MAPE", scores['MAPE'], epoch)

            # writer.add_scalars(f'score/{phase}', scores, global_step=epoch)
            print(scores)
            if phase == 'val' and scores['RMSE'] < worst_rmse:
                worst_rmse = scores['RMSE']
                save_dict.update(model_state_dict=copy.deepcopy(model.state_dict()),
                                 epoch=epoch,
                                 optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))

        scheduler.step()

        # writer.add_scalars('Loss', {f'{phase} loss': running_loss[phase] / len(data_loaders[phase].dataset) for phase in phases}, global_step=epoch)

    writer.flush()
    writer.close()
