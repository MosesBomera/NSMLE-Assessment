import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from typing import List, Optional

class Evaluator:
    """
    A class for evaluating a trained scikit-learn classifier on training and test sets.

    Parameters
    ----------
    model
        A trained scikit-learn classifier.
    X_train : pd.DataFrame or np.ndarray
        Training feature set.
    X_test : pd.DataFrame or np.ndarray
        Test feature set.
    y_train : pd.Series or np.ndarray
        Training target values.
    y_test : pd.Series or np.ndarray
        Test target values.
    df_train : pd.DataFrame
        Training DataFrame containing features and target.
    df_test : pd.DataFrame
        Test DataFrame containing features and target.
    
    Attributes
    ----------
    df_train : pd.DataFrame
        Training DataFrame with predictions.
    df_test : pd.DataFrame
        Test DataFrame with predictions.
    train_metrics : dict
        Dictionary of evaluation metrics for the training set.
    test_metrics : dict
        Dictionary of evaluation metrics for the test set.

    Methods
    -------
    evaluate(verbose=True):
        Evaluate the model and append predictions to self.df_train and self.df_test.
    
    from_dataframe(cls, model, df_train, df_test, features, target):
        Class method to initialize the evaluator from pandas DataFrame inputs.
    """

    def __init__(
        self, 
        model, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame,
        y_train: pd.Series, 
        y_test: pd.Series, 
        df_train: pd.DataFrame, 
        df_test: pd.DataFrame
    ) -> None:
        """
        Initialize the Evaluator with the model and datasets.
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.df_train = df_train.copy()  # Copy DataFrames to avoid modifying original data
        self.df_test = df_test.copy()

        self.train_metrics = {}
        self.test_metrics = {}

    def evaluate(self, verbose: Optional[bool] = False) -> None:
        """
        Evaluate the model on both the train and test sets, appending predictions 
        to the respective DataFrames (self.df_train and self.df_test), and optionally 
        print precision, recall, F1-score, and confusion matrix.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints out evaluation metrics. Default is False.

        Returns
        -------
        None
        """
        # Predictions on training and test data
        self.df_train['predictions'] = self.model.predict(self.X_train)
        self.df_test['predictions'] = self.model.predict(self.X_test)

        # Calculate and store metrics for training data
        self.train_metrics = self._report_metrics(self.y_train, self.df_train['predictions'], verbose=verbose, data_type="Training")

        # Calculate and store metrics for test data
        self.test_metrics = self._report_metrics(self.y_test, self.df_test['predictions'], verbose=verbose, data_type="Test")

    def _report_metrics(self, y_true: pd.Series, y_pred: pd.Series, verbose: Optional[bool] = True, data_type: str = "Training") -> dict:
        """
        Helper function to calculate precision, recall, F1-score, and confusion matrix.

        Parameters
        ----------
        y_true : pd.Series
            True target values.
        y_pred : pd.Series
            Predicted target values.
        verbose : bool, optional
            If True, prints the metrics.
        data_type : str, optional
            Indicates whether the metrics are for training or test data. Default is "Training".

        Returns
        -------
        metrics : dict
            Dictionary containing precision, recall, F1 score, and confusion matrix.
        """
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm
        }

        if verbose:
            print(f"{data_type} Evaluation:")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print("Confusion Matrix:")
            print(cm)

        return metrics

    @classmethod
    def from_dataframe(
        cls, 
        model, 
        df_train: pd.DataFrame, 
        df_test: pd.DataFrame,
        features: List[str], 
        target: str
    ) -> "Evaluator":
        """
        Class method to initialize the Evaluator from pandas DataFrame inputs.

        Parameters
        ----------
        model : ClassifierMixin
            A trained scikit-learn classifier.
        df_train : pd.DataFrame
            Training DataFrame containing both features and target.
        df_test : pd.DataFrame
            Test DataFrame containing both features and target.
        features : List[str]
            List of feature column names.
        target : str
            Name of the target column.

        Returns
        -------
        Evaluator
            An instance of the Evaluator class.

        Raises
        ------
        ValueError
            If the inputs are not valid.
        """
        # Validate the inputs
        if not isinstance(df_train, pd.DataFrame) or not isinstance(df_test, pd.DataFrame):
            raise ValueError("df_train and df_test must be pandas DataFrames.")
        
        if target not in df_train.columns or target not in df_test.columns:
            raise ValueError(f"The target column '{target}' must exist in both df_train and df_test.")
        
        if not set(features).issubset(df_train.columns) or not set(features).issubset(df_test.columns):
            raise ValueError("Some of the feature columns are missing in the DataFrames.")

        # Extract the features and target columns
        X_train = df_train[features]
        X_test = df_test[features]
        y_train = df_train[target]
        y_test = df_test[target]

        return cls(model, X_train, X_test, y_train, y_test, df_train, df_test)