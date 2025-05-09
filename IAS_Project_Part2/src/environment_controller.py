import torch
import numpy as np
import pandas as pd
from .lstm_model import MultiOutputLSTMModel
from datetime import timedelta


class EnvironmentController:
    """EnvironmentController is a class designed to manage and simulate the indoor environment
    conditions (temperature and humidity) based on outdoor conditions and control mechanisms
    such as mechanical ventilation and storage heaters. It supports predictive modeling using
    LSTM, Random Forest, or XGBoost models.
    Attributes:
        feature_names (list): List of input feature names for the model.
        scaler (object): Scaler object for preprocessing input data (optional).
        model_type (str): Type of model used for predictions ('lstm', 'rf', or 'xgb').
        device (torch.device): Device used for LSTM model ('cuda' or 'cpu').
        input_size (int): Number of input features for the LSTM model.
        model (torch.nn.Module): LSTM model for predictions.
        temp_model (object): Model for predicting temperature (used for 'rf' or 'xgb').
        humidity_model (object): Model for predicting humidity (used for 'rf' or 'xgb').
        mech_vent_active (bool): Indicates if mechanical ventilation is active.
        mech_vent_end_time (datetime): End time for mechanical ventilation.
        heater_active (bool): Indicates if storage heaters are active.
        heater_end_time (datetime): End time for storage heater operation.
        heater_uses_today (int): Number of times heaters have been used today.
        last_date (datetime.date): Last date when the heater usage was tracked.
        history (list): List to store historical indoor conditions.
    Methods:
        __init__(input_features, model_type="lstm", model_path=None, temp_model=None,
                humidity_model=None, scaler=None):
            Initializes the EnvironmentController with the specified model and parameters.
        predict_indoor_conditions(input_data):
            Predicts indoor temperature and humidity based on the input data.
        apply_mechanical_ventilation(indoor_temp, indoor_humidity, outdoor_temp,
                                    outdoor_humidity, current_time):
            Simulates the effects of mechanical ventilation on indoor conditions.
        apply_storage_heaters(indoor_temp, current_time):
            Simulates the effects of storage heaters on indoor temperature.
        activate_mechanical_ventilation(current_time):
            Activates mechanical ventilation for a 15-minute period.
        activate_storage_heaters(current_time):
            Activates storage heaters for a 4-hour period, if not already active and
            daily usage limit is not reached.
        control_environment(outdoor_data, start_time, num_intervals, output_file):
            Main control function to process outdoor data, simulate indoor conditions,
            and apply control mechanisms over a specified number of intervals. Saves
            the results to a CSV file."""

    def __init__(
        self,
        input_features,
        model_type="lstm",
        model_path=None,
        temp_model=None,
        humidity_model=None,
        scaler=None,
    ):
        # Set feature names
        self.feature_names = input_features
        self.scaler = scaler
        self.model_type = model_type.lower()

        # LSTM setup
        if self.model_type == "lstm":
            # Load model
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.input_size = len(input_features)
            self.model = MultiOutputLSTMModel(input_size=self.input_size).to(
                self.device
            )
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        # Random Forest or XGBoost setup
        elif self.model_type in ["rf", "xgb"]:
            self.temp_model = temp_model
            self.humidity_model = humidity_model
        else:
            raise ValueError("Invalid model_type. Choose 'lstm', 'rf', or 'xgb'")

        # Control parameters
        self.mech_vent_active = False
        self.mech_vent_end_time = None
        self.heater_active = False
        self.heater_end_time = None
        self.heater_uses_today = 0
        self.last_date = None

        # Store conditions history
        self.history = []

    def predict_indoor_conditions(self, input_data):
        """Use the model to predict indoor temperature and humidity"""
        if self.model_type == "lstm":
            input_tensor = torch.FloatTensor([input_data]).unsqueeze(1).to(self.device)

            with torch.no_grad():
                temp_pred, humidity_pred = self.model(input_tensor)
                temp_pred = temp_pred.item()
                humidity_pred = humidity_pred.item()

        elif self.model_type in ["rf", "xgb"]:
            # Reshape input for sklearn models
            input_array = np.array([input_data])

            # Predict temperature and humidity
            temp_pred = self.temp_model.predict(input_array)[0]
            humidity_pred = self.humidity_model.predict(input_array)[0]

        return temp_pred, humidity_pred

    def apply_mechanical_ventilation(
        self, indoor_temp, indoor_humidity, outdoor_temp, outdoor_humidity, current_time
    ):
        """Apply mechanical ventilation effects"""
        if self.mech_vent_active:
            # If mechanical ventilation is active, blend indoor and outdoor conditions
            if current_time < self.mech_vent_end_time:
                # During the 15-minute ventilation period, indoor conditions match outdoor conditions
                indoor_temp = outdoor_temp
                indoor_humidity = outdoor_humidity
            else:
                # Ventilation period has ended
                self.mech_vent_active = False

        return indoor_temp, indoor_humidity

    def apply_storage_heaters(self, indoor_temp, current_time):
        """Apply storage heater effects"""
        # Check if we need to reset daily heater uses
        current_date = current_time.date()
        if self.last_date is not None and current_date != self.last_date:
            self.heater_uses_today = 0
        self.last_date = current_date

        if self.heater_active:
            # If heaters are active and we're within the 4-hour period
            if current_time < self.heater_end_time:
                if (
                    not self.mech_vent_active
                ):  # Only apply heating if ventilation is off
                    indoor_temp += 0.5  # Increase by 0.5°C per 15 minutes
            else:
                # 4-hour heating period has ended
                self.heater_active = False

        return indoor_temp

    def activate_mechanical_ventilation(self, current_time):
        """Activate mechanical ventilation for 15 minutes"""
        if not self.mech_vent_active:
            self.mech_vent_active = True
            self.mech_vent_end_time = current_time + timedelta(minutes=15)
            return True
        return False

    def activate_storage_heaters(self, current_time):
        """Activate storage heaters for 4 hours if not already active and limit not reached"""
        if not self.heater_active and self.heater_uses_today < 2:
            self.heater_active = True
            self.heater_end_time = current_time + timedelta(hours=4)
            self.heater_uses_today += 1
            return True
        return False

    def control_environment(self, outdoor_data, start_time, num_intervals, output_file):
        """
        Main control function to process outdoor data and control indoor environment

        Parameters:
        - outdoor_data: DataFrame with outdoor conditions
        - start_time: datetime object for the start time
        - num_intervals: number of 15-minute intervals to process
        - output_file: path to output CSV file
        """
        current_time = start_time
        results = []

        # Initialize indoor conditions with predictions or default values
        if len(outdoor_data) > 0:
            initial_features = [
                outdoor_data.iloc[0][feature] for feature in self.feature_names
            ]
            indoor_temp, indoor_humidity = self.predict_indoor_conditions(
                initial_features
            )
        else:
            # Default values if no outdoor data
            indoor_temp, indoor_humidity = 20.0, 50.0

        for interval in range(num_intervals):
            # Get closest outdoor data point
            outdoor_idx = min(interval, len(outdoor_data) - 1)
            outdoor_row = outdoor_data.iloc[outdoor_idx]
            outdoor_temp = outdoor_row.get(
                "Outside temp", 15.0
            )  # Default if not in data
            outdoor_humidity = outdoor_row.get(
                "Outdoor_relative_humidity_Sensor", 50.0
            )  # Default if not in data

            # Create feature vector for prediction
            features = [
                outdoor_row[feature] if feature in outdoor_row else 0
                for feature in self.feature_names
            ]

            # Decision making for environmental control
            # Example: ventilate every 6 hours
            if interval % 24 == 0:
                self.activate_mechanical_ventilation(current_time)

            # Example: heat if much colder outside (twice a day)
            if interval % 48 == 0 and outdoor_temp < indoor_temp - 2:
                self.activate_storage_heaters(current_time)

            # Apply active control effects
            indoor_temp, indoor_humidity = self.apply_mechanical_ventilation(
                indoor_temp,
                indoor_humidity,
                outdoor_temp,
                outdoor_humidity,
                current_time,
            )
            indoor_temp = self.apply_storage_heaters(indoor_temp, current_time)

            # Predict next state
            next_features = features.copy()
            # Update features with new indoor conditions
            if "Indoor_temperature_room" in self.feature_names:
                idx = self.feature_names.index("Indoor_temperature_room")
                next_features[idx] = indoor_temp
            if "Humidity" in self.feature_names:
                idx = self.feature_names.index("Humidity")
                next_features[idx] = indoor_humidity

            # Make prediction for next interval
            new_temp, new_humidity = self.predict_indoor_conditions(next_features)

            # Blend prediction with control effects (weighted average)
            indoor_temp = indoor_temp * 0.7 + new_temp * 0.3
            indoor_humidity = indoor_humidity * 0.7 + new_humidity * 0.3

            # Record the state
            state = {
                "DateTime": current_time.strftime("%Y-%m-%d %H:%M"),
                "Indoor_temperature": indoor_temp,
                "Indoor_humidity": indoor_humidity,
                "Outdoor_temperature": outdoor_temp,
                "Outdoor_humidity": outdoor_humidity,
                "Mechanical_ventilation": 1 if self.mech_vent_active else 0,
                "Storage_heaters": 1 if self.heater_active else 0,
            }
            results.append(state)

            # Move to next interval (15 minutes)
            current_time += timedelta(minutes=15)

        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(
            f"Environment control simulation complete. Results saved to {output_file}"
        )
        return results_df
