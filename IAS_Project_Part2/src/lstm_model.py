import torch.nn as nn


class MultiOutputLSTMModel(nn.Module):
    """
    A PyTorch module for a multi-output LSTM model that predicts temperature and humidity
    from sequential input data. The model includes an LSTM layer, batch normalization,
    and separate fully connected layers for each output.
    Attributes:
        hidden_size (int): The number of features in the hidden state of the LSTM.
        lstm (nn.LSTM): The LSTM layer with specified input size, hidden size, number of layers,
                        and dropout.
        batch_norm (nn.BatchNorm1d): Batch normalization layer applied to the LSTM output.
        fc_temp (nn.Linear): Fully connected layer for temperature prediction.
        fc_humidity (nn.Linear): Fully connected layer for humidity prediction.
    Args:
        input_size (int): The number of input features for the LSTM.
        hidden_size (int, optional): The number of features in the hidden state of the LSTM.
                                     Default is 64.
        num_layers (int, optional): The number of recurrent layers in the LSTM. Default is 2.
        dropout (float, optional): Dropout probability for the LSTM layers. Default is 0.2.
    Methods:
        forward(x):
            Performs a forward pass through the model.
            Args:
                x (torch.Tensor): Input tensor of shape [batch, seq_len, features].
            Returns:
                tuple: A tuple containing:
                    - temp_pred (torch.Tensor): Predicted temperature values of shape [batch, 1].
                    - humidity_pred (torch.Tensor): Predicted humidity values of shape [batch, 1].
    """

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(MultiOutputLSTMModel, self).__init__()
        self.hidden_size = hidden_size

        # LSTM with dropout between layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # Add batch normalization for more stable training
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        self.fc_temp = nn.Linear(hidden_size, 1)
        self.fc_humidity = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: [batch, seq_len, features]
        lstm_out, _ = self.lstm(x)

        # Take only the last time step output
        lstm_out = lstm_out[:, -1, :]

        # Apply batch normalization
        normalized = self.batch_norm(lstm_out)

        # Output layers
        temp_pred = self.fc_temp(normalized)
        humidity_pred = self.fc_humidity(normalized)

        return temp_pred, humidity_pred
