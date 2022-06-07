from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils.util import get_loss


def evaluate(predictions, targets, normal):
    normal = normal[0]
    n_samples = targets.shape[0]
    scores = defaultdict(dict)
    for horizon in range(12):
        y_true = np.reshape(targets[:, horizon], (n_samples, -1))
        y_pred = np.reshape(predictions[:, horizon], (n_samples, -1))
        scores['MAE'][f'horizon-{horizon}'] = normal.mae_transform(mean_absolute_error(y_pred, y_true))
        scores['RMSE'][f'horizon-{horizon}'] = normal.rmse_transform(np.sqrt(mean_squared_error(y_pred, y_true)))
        scores['PCC'][f'horizon-{horizon}'] = pcc(y_pred, y_true)
    y_true = np.reshape(targets, (n_samples, -1))
    y_pred = np.reshape(predictions, (n_samples, -1))
    scores['rmse'] = normal.rmse_transform(np.sqrt(mean_squared_error(y_true, y_pred)))
    scores['mae'] = normal.mae_transform(mean_absolute_error(y_pred, y_true))
    scores['pcc'] = pcc(y_pred, y_true)
    return scores

def pcc(x, y):
    x, y = x.reshape(-1), y.reshape(-1)
    return np.corrcoef(x, y)[0][1]


def mask_evaluate(predictions: np.ndarray, targets: np.ndarray):
    assert targets.shape == predictions.shape and targets.shape[1] == 12, f'{targets.shape}/{predictions.shape}'
    n_samples = targets.shape[0]
    scores = defaultdict(dict)
    for horizon in range(12):
        y_true = np.reshape(targets[:, horizon], (n_samples, -1))
        y_pred = np.reshape(predictions[:, horizon], (n_samples, -1))
        scores['masked MAE'][f'horizon-{horizon}'] = masked_mae_np(y_pred, y_true, null_val=0.0)
        scores['masked RMSE'][f'horizon-{horizon}'] = masked_rmse_np(y_pred, y_true, null_val=0.0)
        scores['masked MAPE'][f'horizon-{horizon}'] = masked_mape_np(y_pred, y_true, null_val=0.0) * 100.0

    return scores


def nomask_evaluate(predictions, targets):
    assert targets.shape == predictions.shape and targets.shape[1] == 12, f'{targets.shape}/{predictions.shape}'
    predictions = torch.from_numpy(np.asarray(predictions)).float()
    targets = torch.from_numpy(np.asarray(targets)).float()
    n_samples = targets.shape[0]
    scores = defaultdict(dict)
    loss = get_loss('rmse')
    for horizon in range(12):
        y_true = np.reshape(targets[:, horizon], (n_samples, -1))
        y_pred = np.reshape(predictions[:, horizon], (n_samples, -1))
        # scores['masked MAE'][f'horizon-{horizon}'] = masked_mae_np(y_pred, y_true, null_val=0.0)
        scores['masked RMSE'][f'horizon-{horizon}'] = loss(y_true, y_pred).cpu().numpy()
        # scores['masked MAPE'][f'horizon-{horizon}'] = masked_mape_np(y_pred, y_true, null_val=0.0) * 100.0
    scores['loss'] = loss(targets, predictions).cpu().numpy()
    return scores


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float64')
        mask /= np.mean(mask)
        mse = np.square(np.subtract(preds, labels)).astype('float64')
        mse = np.nan_to_num(mse * mask)
        return np.mean(mse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float64')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float64')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float64')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float64'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)
