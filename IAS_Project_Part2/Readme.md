# Indoor Environment Prediction and Control System

## Project Overview

This project focuses on developing intelligent adaptive systems for predicting and controlling indoor environmental conditions (temperature and humidity) using various machine learning approaches. The system analyzes environmental sensor data to build predictive models that can optimize indoor comfort levels while considering external conditions.

## Key Features

- Data exploration and analysis of indoor environmental parameters
- Feature selection and preprocessing for optimal model performance
- Implementation of multiple machine learning models:
  - LSTM neural networks with multi-output capabilities
  - Random Forest regression
  - XGBoost regression
- Environment simulation for testing model performance
- Comparative analysis of different model approaches

## Project Structure

- **Data Loading**: Import and initial examination of environmental sensor data
- **EDA (Exploratory Data Analysis)**: Statistical analysis and visualization of key patterns
- **Data Preparation & Preprocessing**: Handling missing values, outlier detection, and feature engineering
- **Feature Selection**: Identifying the most relevant features for prediction models
- **LSTM Model Building**: Development of deep learning model with PyTorch
- **Other Models**: Implementation of Random Forest and XGBoost alternatives
- **Environment Simulation**: Testing model performance in simulated conditions

## Data Description

The dataset contains time-series environmental measurements including:

- Indoor temperature and humidity
- CO2 levels
- Outside temperature and humidity
- Lighting conditions
- Occupancy information
- Meteorological data (rain, wind, sun exposure)

## Model Results

The LSTM, Random Forest, and XGBoost models were compared across multiple performance metrics:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

## Requirements

The project requires the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- PyTorch
- XGBoost
- SciPy

## Usage

1. Ensure all required libraries are installed
2. Run the notebook to:
   - Load and explore data
   - Preprocess environmental measurements
   - Train different models
   - Test prediction performance
   - Visualize results

## Visualizations

The project includes multiple visualizations:

- Correlation heatmaps of environmental variables
- Time series plots of temperature and humidity
- Distribution plots of key variables
- Feature importance comparisons
- Model performance comparisons

## Author

Jasser Abdelfattah