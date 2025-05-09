import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_with_multiple_metrics(model, test_loader):
    """
    Evaluates a given model using multiple regression metrics for two outputs: temperature and humidity.
    Args:
        model (torch.nn.Module): The trained model to evaluate. It should output predictions for temperature and humidity.
        test_loader (torch.utils.data.DataLoader): DataLoader containing the test dataset. Each batch should return
            inputs and targets, where targets is a tensor with temperature in the first column and humidity in the second column.
    Returns:
        tuple: A tuple containing two dictionaries:
            - temp_metrics (dict): Metrics for temperature predictions, including:
                - "MSE": Mean Squared Error
                - "RMSE": Root Mean Squared Error
                - "MAE": Mean Absolute Error
                - "R²": Coefficient of Determination (R-squared)
                - "MAPE": Mean Absolute Percentage Error
            - humidity_metrics (dict): Metrics for humidity predictions, including:
                - "MSE": Mean Squared Error
                - "RMSE": Root Mean Squared Error
                - "MAE": Mean Absolute Error
                - "R²": Coefficient of Determination (R-squared)
                - "MAPE": Mean Absolute Percentage Error
    """

    model.eval()
    temp_preds, temp_targets = [], []
    humidity_preds, humidity_targets = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            temp_out, humidity_out = model(inputs)

            temp_preds.extend(temp_out.cpu().numpy())
            temp_targets.extend(targets[:, 0:1].cpu().numpy())

            humidity_preds.extend(humidity_out.cpu().numpy())
            humidity_targets.extend(targets[:, 1:2].cpu().numpy())

    # Convert to numpy arrays
    temp_preds = np.array(temp_preds).flatten()
    temp_targets = np.array(temp_targets).flatten()
    humidity_preds = np.array(humidity_preds).flatten()
    humidity_targets = np.array(humidity_targets).flatten()

    # Calculate metrics for temperature
    temp_metrics = {
        "MSE": mean_squared_error(temp_targets, temp_preds),
        "RMSE": np.sqrt(mean_squared_error(temp_targets, temp_preds)),
        "MAE": mean_absolute_error(temp_targets, temp_preds),
        "R²": r2_score(temp_targets, temp_preds),
        "MAPE": np.mean(np.abs((temp_targets - temp_preds) / temp_targets)) * 100,
    }

    # Calculate metrics for humidity
    humidity_metrics = {
        "MSE": mean_squared_error(humidity_targets, humidity_preds),
        "RMSE": np.sqrt(mean_squared_error(humidity_targets, humidity_preds)),
        "MAE": mean_absolute_error(humidity_targets, humidity_preds),
        "R²": r2_score(humidity_targets, humidity_preds),
        "MAPE": np.mean(np.abs((humidity_targets - humidity_preds) / humidity_targets))
        * 100,
    }

    return temp_metrics, humidity_metrics
