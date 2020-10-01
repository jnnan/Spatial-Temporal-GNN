import torch


def calculate_scores(loss_funcs, predict, truth, scaler):
    scores = {}
    scores['MAE'] = loss_funcs['masked_mae_loss'](predict=predict, truth=truth, scaler=scaler)
    scores['MAPE'] = loss_funcs['masked_mape_loss'](predict=predict, truth=truth, scaler=scaler)
    scores['RMSE'] = loss_funcs['masked_rmse_loss'](predict=predict, truth=truth, scaler=scaler)
    return scores


def create_loss():

    def masked_mse_loss(predict, truth, scaler):
        predict = scaler.inverse_transform(predict)
        truth = scaler.inverse_transform(truth)
        mask = truth != 0.0
        predict = predict[mask]
        truth = truth[mask]
        return torch.mean(torch.square(predict - truth))

    def masked_rmse_loss(predict, truth, scaler):
        predict = scaler.inverse_transform(predict)
        truth = scaler.inverse_transform(truth)
        mask = truth != 0.0
        predict = predict[mask]
        truth = truth[mask]
        return torch.sqrt(torch.mean(torch.square(predict - truth)))

    def masked_mape_loss(predict, truth, scaler):
        predict = scaler.inverse_transform(predict)
        truth = scaler.inverse_transform(truth)
        mask = truth != 0.0
        predict = predict[mask]
        truth = truth[mask]
        return torch.mean(torch.abs(torch.true_divide(predict - truth, truth)))

    def masked_mae_loss(predict, truth, scaler):
        predict = scaler.inverse_transform(predict)
        truth = scaler.inverse_transform(truth)
        mask = truth != 0.0
        predict = predict[mask]
        truth = truth[mask]
        return torch.mean(torch.abs(predict - truth))

    loss_dict = {}
    loss_dict['masked_mse_loss'] = masked_mse_loss
    loss_dict['masked_rmse_loss'] = masked_rmse_loss
    loss_dict['masked_mape_loss'] = masked_mape_loss
    loss_dict['masked_mae_loss'] = masked_mae_loss
    return loss_dict
