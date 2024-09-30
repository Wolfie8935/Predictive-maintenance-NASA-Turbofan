# Predictive Maintenance for NASA Turbofan Engines

## Project Overview

This project focuses on **predicting the Remaining Useful Life (RUL)** of turbofan engines using the NASA CMAPSS dataset. We aim to build a predictive model to forecast engine failure, which enables proactive maintenance scheduling and reduces operational risks.

The project is structured in two approaches:
1. **Approach 1**: Direct prediction of RUL using regression models.
2. **Approach 2**: Clipped RUL values to improve accuracy, especially for engines close to failure.

We have completed the entire workflow, from **data cleaning and feature engineering to model training and testing**, using the data from the `_FD001.txt` file. Future work will extend the analysis to the remaining datasets (`_FD002.txt`, `_FD003.txt`, `_FD004.txt`).

## Dataset

The dataset is part of the **NASA CMAPSS Turbofan Engine Degradation** simulation dataset, specifically focusing on `_FD001.txt`. The data can be downloaded from the official source here: [NASA CMAPSS Dataset](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6/about_data)

### Data Files Used:
- `train_FD001.txt`: Training dataset for engine degradation.
- `test_FD001.txt`: Test dataset for validation.
- `RUL_FD001.txt`: True Remaining Useful Life values for test set engines.

## Steps Involved

### 1. Data Cleaning and Preprocessing
- Filtered and cleaned data for consistent formatting.
- Removed redundant columns and handled missing values (if any).
- Created engine-wise data slices for training.

### 2. Exploratory Data Analysis (EDA)
- Performed analysis to understand the trends in engine degradation.
- Visualized sensor readings and operational settings for feature selection.

### 3. Feature Engineering and Extraction
- Engineered key features based on domain knowledge.
- Added rolling averages for smoother trend analysis.
- Normalized data to improve model performance.

### 4. Modeling
We experimented with different regression models to predict the RUL:

- **Linear Regression**
- **Decision Tree**
- **Random Forest**
- **XGBoost**
  
After evaluating Approach 1 and Approach 2, the models were trained and tuned on the `train_FD001.txt` dataset and validated on the `test_FD001.txt` dataset.

### 5. Training and Fine-Tuning
We trained and fine-tuned each model on the training data, testing various hyperparameters to improve accuracy.

### 6. Approach 2: Clipping the RUL to 125
In Approach 2, we clipped the RUL values to a maximum threshold, focusing the model's prediction on engines close to failure. This improved the model's performance, particularly for engines nearing end-of-life.

## Results

### Approach 1: Direct RUL Prediction
The following models were evaluated using the `train_FD001.txt` dataset:

| Model             | Dataset    | RMSE  | R² Score  |
|-------------------|------------|-------|-----------|
| **Linear Regression** | Train      | 44.82 | 58.05%    |
|                   | Test       | 44.29 | 57.06%    |
|                   | Validation | 31.87 | 41.18%    |
| **Decision Tree**  | Train      | 0.00  | 100.00%   |
|                   | Test       | 60.07 | 21.03%    |
|                   | Validation | 45.89 | -21.95%   |
| **Random Forest**  | Train      | 15.77 | 94.80%    |
|                   | Test       | 41.37 | 62.54%    |
|                   | Validation | 31.83 | 41.34%    |
| **SVR**           | Train      | 42.91 | 61.54%    |
|                   | Test       | 42.25 | 60.93%    |
|                   | Validation | 26.15 | 60.41%    |
| **XGBoost**       | Train      | 40.50 | 65.75%    |
|                   | Test       | 41.77 | 61.81%    |
|                   | Validation | 29.79 | 48.60%    |

### Approach 2: Clipping the RUL

In Approach 2, we clipped the RUL values to improve the model's ability to predict when engines were closer to failure.

| Model             | Dataset    | RMSE  | R² Score  |
|-------------------|------------|-------|-----------|
| **Random Forest**  | Validation | 17.99 | 81.26%    |
| **SVR**           | Validation | 19.63 | 77.69%    |
| **XGBoost**       | Validation | 18.18 | 80.87%    |


### Summary of Results
- **Approach 1**: The highest accuracy in Approach 1 was obtained with **Random Forest** and **XGBoost**, with R² scores of around **41-48%** for the validation set.
- **Approach 2 (Clipping)**: The performance significantly improved with **Random Forest** and **XGBoost**, achieving **81.26%** and **80.87%** R² scores on the validation set.


## Future Work
- Extend the analysis to `_FD002.txt`, `_FD003.txt`, and `_FD004.txt` for further validation and testing.
- Experiment with advanced machine learning techniques like deep learning for enhanced prediction accuracy.

## Conclusion

This project demonstrated the power of machine learning in **predictive maintenance** for NASA turbofan engines. We explored different models and approaches, achieving a significant improvement in accuracy with the second approach by clipping RUL values. Future work will continue to refine the models on additional datasets.
