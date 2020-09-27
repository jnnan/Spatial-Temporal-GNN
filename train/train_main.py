import os
import shutil
# import time
import torch
import copy
from torch import optim
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils.metric import calculate_metrics
from data_loader.data_container import load_dataset
from model.dcrnn_model import DCRNNmodel
from utils.util import load_graph_data, create_diffusion_supports, calculate_random_walk_matrix, convert_to_gpu
from utils.loss import masked_rmse_loss, masked_mse_loss, MSELoss, BCELoss


def create_loss(loss_type, **kwargs):
    if loss_type == 'mse_loss':
        return convert_to_gpu(MSELoss())
    elif loss_type == 'bce_loss':
        return convert_to_gpu(BCELoss())
    elif loss_type == 'masked_rmse_loss':
        return masked_rmse_loss(kwargs['scaler'], 0.0)
    elif loss_type == 'masked_mse_loss':
        return masked_mse_loss(kwargs['scaler'], 0.0)
    else:
        raise ValueError("Unknown loss function.")


def train_main(args):
    save_name = args['save_name']
    model_name = args['model_name']
    model_folder = f"data/save_models/{model_name}+{save_name}"
    tensorboard_folder = f"runs/{model_name}+{save_name}"
    print(f'train {model_name} {save_name}')

    data_loaders, scaler = load_dataset(args)

    _, _, adj = load_graph_data(args['adj_dir'])
    K = args['max_diffusion_step']
    kernals = []
    kernals.append(calculate_random_walk_matrix(adj).T)
    kernals.append(calculate_random_walk_matrix(adj.T).T)
    supports = create_diffusion_supports(kernals, K, args['num_nodes'])

    model = convert_to_gpu(DCRNNmodel(supports, args))

    loss_func = create_loss(loss_type=args['loss_function'], scaler=scaler)
    if os.path.exists(model_folder):
        shutil.rmtree(model_folder)
    if os.path.exists(tensorboard_folder):
        shutil.rmtree(tensorboard_folder)
    os.makedirs(model_folder, exist_ok=True)
    os.makedirs(tensorboard_folder, exist_ok=True)
    optimizer = optim.Adam(model.parameters(),
                           lr=args['learning_rate'],
                           weight_decay=args['weight_decay'])

    phases = ['train', 'val', 'test']
    writer = SummaryWriter(tensorboard_folder)
    # since = time.clock()
    # loss_func = convert_to_gpu(loss_func)  # Todo
    save_dict, worst_rmse = {'model_state_dict': copy.deepcopy(model.state_dict()), 'epoch': 0}, 100000
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.2, patience=5, threshold=1e-3, min_lr=1e-6)

    kwargs = {}
    kwargs['scaler'] = scaler

    for epoch in range(args['num_epochs']):
        running_loss = {phase: 0.0 for phase in phases}

        for phase in phases:
            if phase == 'train':
                kwargs['is_eval'] = False
            else:
                kwargs['is_eval'] = True
            steps, predictions, targets = 0, list(), list()
            tqdm_loader = tqdm(enumerate(data_loaders[phase]))
            # ite = 0
            for step, (features, truth_data) in tqdm_loader:
                # ite = ite + 1
                # if ite > 10:
                #     break
                features = convert_to_gpu(features)
                truth_data = convert_to_gpu(truth_data)
                global_step = (epoch) * 2500 + steps/10
                kwargs['global_step'] = global_step
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(features, **kwargs)
                    outputs = outputs.squeeze()
                    loss = loss_func(truth=truth_data, predict=outputs)
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                targets.append(truth_data.cpu().numpy())
                with torch.no_grad():
                    predictions.append(outputs.cpu().numpy())
                running_loss[phase] += loss * truth_data.size(0)
                steps += truth_data.size(0)

                tqdm_loader.set_description(
                    f'{phase} epoch: {epoch}, {phase} loss: {running_loss[phase] / steps}')

                torch.cuda.empty_cache()

            predictions = np.concatenate(predictions)
            targets = np.concatenate(targets)
            # print(predictions[:3, :3])
            # print(targets[:3, :3])
            scores = calculate_metrics(predictions.reshape(predictions.shape[0], -1), targets.reshape(targets.shape[0], -1), **kwargs)
            writer.add_scalars(f'score/{phase}', scores, global_step=epoch)
            print(scores)
            if phase == 'val' and scores['RMSE'] < worst_rmse:
                worst_rmse = scores['RMSE'],
                save_dict.update(model_state_dict=copy.deepcopy(model.state_dict()),
                                 epoch=epoch,
                                 optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))

        scheduler.step(running_loss['train'])

        writer.add_scalars('Loss', {
            f'{phase} loss': running_loss[phase] / len(data_loaders[phase].dataset) for phase in phases},
                            global_step=epoch)
