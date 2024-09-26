## Notes
- `description_of_approach.md` contains a high level description of the modelling approach.
- Notebook `002_How_To_Train_Model.ipynb` contains a step by step guide on how to train the model.
- Notebook `001_Exploratory_Data_Analysis.ipynb` contains the approach I took for the exploratory data analysis.
- `requirements.txt` contains the packages to install to run training modules.

### Training utilities in `train_utils` folder:

1. **`feature_engineering.py`**:
   - **Purpose**: Contains functions for preprocessing and feature extraction. This includes transforming raw data into meaningful features used for model training.
   - **Key Functions**: 
     - `preprocess_dataframe()`: Cleans and prepares raw data.
     - `extract_historical_loan_features()`: Extracts features from the historical loan data.
     - `calculate_loan_repayment_features()`: Creates repayment-related features from the repayment data.
   - **Usage**: Import these functions to preprocess raw data and extract features for model training.

2. **`train.py`**:
   - **Purpose**: Defines the `Train` class responsible for orchestrating the training process, including model fitting and cross-validation.
   - **Key Features**:
     - Implements cross-validation, parameter tuning, and model training with custom scoring.
   - **Usage**: Import and instantiate the `Train` class to handle the training workflow, passing an estimator (model), grid search parameters, training data, features and target name etc.

3. **`eval.py`**:
   - **Purpose**: Provides the `Evaluator` class, used for evaluating the trained classifier model on both training and test data.
   - **Key Functions**:
     - `evaluate()`: Computes evaluation metrics including precision, recall, F1-score, and confusion matrix.
     - `from_dataframe()`: A convenient method to initialize evaluation from DataFrames.
   - **Usage**: Use this class after model training to evaluate the performance on different datasets.

4. **`run_training.py`**:
   - **Purpose**: Main script that orchestrates the entire model training process. It handles preprocessing, model setup, and evaluation.
   - **Key Functions**:
     - `run_training()`: Trains the model, performs cross-validation, and evaluates it.
   - **Usage**: Call `run_training()` with the appropriate data, features, and model configuration to train and evaluate a machine learning model.
  
