import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from typing import Optional, Union

class Train:
    """
    Scikit-Learn estimator training wrapper using GridSearchCV.

    Parameters
    ----------
    estimator
        An estimator that implements the scikit-learn estimator interface.
    estimator_params : dict
        A dictionary with parameters names (str) as keys and lists of parameter settings to try as values,
        or a list of such dictionaries, in which case the grids spanned by each dictionary in the list are
        explored. This enables searching over any sequence of parameter settings.
    feature_data : pd.DataFrame
        A dataframe containing the feature data for estimator fitting.
    feature_names : list
        A list of columns to be used as features for estimator fitting.
    cv : int, CvSplitter
        Determines the cross-validation splitting strategy.
    target : str
        Optionally, name of the column with the target variable. Default is `target`.
    scoring : str, callable, list, tuple or dict
        A str, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on the test set.
    n_jobs : int
        Optionally, the number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors
    fill_nans : bool, False
        Whether to fill nan values with zeros, default is False.
    verbose : bool
        Controls the verbosity: the higher, the more messages.
        >1 : the computation time for each fold and parameter candidate is displayed;
        >2 : the score is also displayed;
        >3 : the fold and candidate parameter indexes are also displayed together with the starting time of the computation.
    """
    def __init__(
        self,
        estimator,
        feature_data: pd.DataFrame,
        feature_names: list,
        estimator_params: dict,
        cv: Union[int, None],
        target: Optional[str] = "target",
        scoring: Union[str, callable, list, tuple, dict] = 'neg_log_loss',
        n_jobs: Optional[int] = -1,
        fill_nans: Optional[bool] = False,
        verbose: Optional[int] = 1
    ):
        """
        Create a Train instance.
        """
        if not isinstance(feature_data, pd.DataFrame):
            raise ValueError("feature_data must be a pandas DataFrame.")

        if feature_data.empty:
            raise ValueError("feature_data DataFrame is empty.")

        if not all(name in feature_data.columns for name in feature_names):
            raise ValueError("feature_names must be a subset of feature_data columns.")

        if target not in feature_data.columns:
            raise ValueError(f"Target column '{target}' not found in feature_data.")

        self.feature_data = feature_data
        self.feature_names = feature_names
        self.target = target
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.fill_nans = fill_nans
        self.verbose = verbose
        self.fitted_estimator = None

        # Train at init.
        self.__call__()

    def __call__(
        self
    ) -> None:
        """
        The training process.
        """
        # Training.
        self.fitted_estimator = self.__train_process()

    def __train_process(
        self
    ) -> GridSearchCV:
        """
        Automates the training pipeline, making the iterative training process
        functionally efficient.

        Returns
        -------
        GridSearchCV
            A fitted GridSearchCV object.
        """
        feature_data = self.feature_data.copy(deep=True)

        if self.fill_nans:
            feature_data = feature_data.replace([np.nan], 0.0)

        # Split the features and labels.
        X, y = feature_data[self.feature_names], feature_data[self.target]

        # The search for the best model.
        model_grid_search = GridSearchCV(
            estimator=self.estimator,
            scoring=self.scoring,
            param_grid=self.estimator_params,
            cv=self.cv,
            verbose=self.verbose,
            return_train_score=True,
            n_jobs=self.n_jobs)

        model_grid_search.fit(X, y)

        return model_grid_search