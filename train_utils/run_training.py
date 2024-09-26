from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import TimeSeriesSplit
from .train import Train
from .eval import Evaluator
import pandas as pd
from typing import Tuple

def train_test_split_by_business_id(
        df: pd.DataFrame, 
        target: str = 'paid_late', 
        train_size: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into train and test sets ensuring that for each business_id,
    only the loan applications in the future are part of the test data.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns including 'business_id', 'application_number',
        'loan_number', and other features.
    target : str, optional
        The name of the target column containing the labels. Default is 'paid_late'.
    train_size : float, optional
        The proportion of the dataset to include in the train split. Default is 0.8.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the train and test dataframes.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    required_columns = {'business_id', 'application_number', 'loan_number', target}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    if not (0 < train_size < 1):
        raise ValueError("train_size must be between 0 and 1.")

    train_list = []
    test_list = []

    # Group by business_id and split the data within each group
    for business_id, group in df.groupby('business_id'):
        # Sort the group by application_number and loan_number
        group = group.sort_values(by=['application_number', 'loan_number'])

        # Calculate the cutoff index based on the train_size
        cutoff_index = int(len(group) * train_size)

        # Split the group into train and test sets
        train_group = group.iloc[:cutoff_index]
        test_group = group.iloc[cutoff_index:]

        # Append the train and test groups to the respective lists
        train_list.append(train_group)
        test_list.append(test_group)

    # Concatenate the lists to form the final train and test dataframes
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    # Ensure the distribution of positive and negative classes is maintained
    train_positive = train_df[train_df[target] == 1]
    train_negative = train_df[train_df[target] == 0]
    test_positive = test_df[test_df[target] == 1]
    test_negative = test_df[test_df[target] == 0]

    train_positive_size = int(len(train_positive) * train_size)
    train_negative_size = int(len(train_negative) * train_size)

    train_df = pd.concat([train_positive.sample(train_positive_size, random_state=42),
                          train_negative.sample(train_negative_size, random_state=42)]).reset_index(drop=True)

    test_df = pd.concat([test_positive, test_negative]).reset_index(drop=True)

    return train_df, test_df

def run_training(
    feature_data: pd.DataFrame, 
    numerical_features: list, 
    categorical_features: list, 
    model, 
    model_params: dict, 
    target: str = 'paid_late',
    train_size: float = 0.8,
    scoring: str = 'neg_log_loss'
):
    """
    Preprocesses data, trains the model, and evaluates it using cross-validation.

    Parameters
    ----------
    feature_data : pd.DataFrame
        DataFrame containing both feature columns and the target column.
    numerical_features : list of str
        List of names of the numerical feature columns.
    categorical_features : list of str
        List of names of the categorical feature columns.
    model : object
        A machine learning model class (e.g., LogisticRegressionCV or RandomForestClassifier).
    model_params : dict
        Dictionary of parameters to pass to the model during training.
    target : str, optional
        The name of the target column. Default is 'paid_late'.
    train_size : float, optional
        The proportion of the dataset to include in the train split. Default is 0.8.
    scoring : str, optional
        Scoring function for cross-validation. Default is 'neg_log_loss'.

    Returns
    -------
    trained_model : object
        The trained machine learning model.
    evaluation_metrics : dict
        The evaluation metrics after model evaluation.
    """
    
    # Combine numerical and categorical features
    features = numerical_features + categorical_features
    
    # Split data into training and test sets
    train, test = train_test_split_by_business_id(feature_data, target, train_size=train_size)

    # Ensure the data is sorted by business_id, application_number, and loan_number
    train = train.sort_values(by=['business_id', 'application_number', 'loan_number']).reset_index(drop=True)

    # Define TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Create preprocessing pipelines for numerical and categorical features
    cat_processor = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore')
    )

    num_processor = make_pipeline(
        SimpleImputer(strategy='median')
    )
    preprocessor = make_column_transformer(
        (cat_processor, categorical_features),
        (num_processor, numerical_features)
    )

    # Build the model pipeline
    estimator = make_pipeline(
        preprocessor,
        model
    )

    # Train the model
    trained_model = Train(
        estimator=estimator,
        feature_data=train,
        feature_names=features,
        target=target,
        estimator_params=model_params,  
        scoring=scoring,
        cv=tscv,
        verbose=1
    )

    # Initialize and evaluate the model
    evaluator = Evaluator.from_dataframe(trained_model.fitted_estimator.best_estimator_, train, test, features, target)
    evaluator.evaluate()

    return trained_model, evaluator