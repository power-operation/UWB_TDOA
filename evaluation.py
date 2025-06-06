import numpy as np

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))

def mae(predictions, targets):
    return np.mean(np.abs(predictions - targets))

def evaluate(predictions, targets):
    """
    Support 2D/3D prediction error evaluation.

    Parameters: 
        predictions: ndarray, shape=(N, 2 or 3) 
        targets: ndarray, shape=(N, 2 or 3)

    Returns: 
        metrics: dict, contains RMSE and MAE overall error 
        per_point_error: ndarray, Euclidean error per valid point 
        valid_mask: ndarray[bool], which points are not labeled as NaNSupport 2D/3D prediction error evaluators.

    Parameters: 
        predictions: ndarray, shape=(N, 2 or 3) 
        targets: ndarray, shape=(N, 2 or 3)

    Returns: 
        metrics: dict with overall RMSE and MAE errors 
        per_point_error: ndarray, Euclidean error per valid point 
        valid_mask: ndarray[bool], which points are not labeled as NaN

    """
    valid_mask = ~np.isnan(predictions).any(axis=1)
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]

    if predictions.shape[0] == 0:
        return {"RMSE": np.nan, "MAE": np.nan}, [], []

    diff = predictions - targets
    per_point_error = np.linalg.norm(diff, axis=1)

    rmse_val = np.sqrt(np.mean(per_point_error ** 2))
    mae_val = np.mean(per_point_error)

    return {"RMSE": rmse_val, "MAE": mae_val}, per_point_error, valid_mask
