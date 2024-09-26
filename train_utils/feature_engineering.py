import pandas as pd
from typing import List
import re

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the DataFrame by cleaning the column names and dropping duplicate rows.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be preprocessed.

    Returns
    -------
    pandas.DataFrame
        A preprocessed DataFrame with cleaned column names and duplicate rows removed.
    """

    def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the column names of a DataFrame by converting to lowercase, removing special characters, 
        and replacing spaces with underscores.
        """
        def clean_name(name: str) -> str:
            # Convert to lowercase
            name = name.lower()
            # Replace special characters with an empty string
            name = re.sub(r'[^a-z0-9\s_]', '', name)
            # Replace spaces with underscores
            name = re.sub(r'\s+', '_', name)
            return name

        df.columns = [clean_name(col) for col in df.columns]
        return df

    def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops duplicate rows from the DataFrame.
        """
        return df.drop_duplicates()

    # Modular steps for preprocessing
    df = clean_column_names(df)  # Clean column names
    df = drop_duplicates(df)     # Drop duplicate rows

    return df

def extract_historical_loan_features(df: pd.DataFrame, known_approval_statuses: List[str]) -> pd.DataFrame:
    """
    Extract historical loan features for each loan application from the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns ['loan_id', 'business_id', 'sector', 'principal',
        'total_owing_at_issue', 'application_number', 'applying_for_loan_number',
        'loan_number', 'employee_count', 'paid_late', 'approval_status'].
    known_approval_statuses : List[str]
        List of known approval statuses to consider (e.g., ['Approved', 'Declined', 'Cancelled', 'Expired']).

    Returns
    -------
    pd.DataFrame
        Dataframe with extracted historical loan features for each loan application.

    Notes
    -----
    The function assumes that the input dataframe is sorted by 'business_id',
    'application_number', and 'loan_number'. The 'paid_late' status of the current
    loan application is excluded from the feature extraction.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    required_columns = {
        'loan_id', 'business_id', 'sector', 'principal', 'total_owing_at_issue',
        'application_number', 'applying_for_loan_number', 'loan_number',
        'employee_count', 'paid_late', 'approval_status'
    }
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    if not isinstance(known_approval_statuses, list) or not all(isinstance(status, str) for status in known_approval_statuses):
        raise TypeError("known_approval_statuses must be a list of strings.")

    # Sort the DataFrame by business_id, application_number, and loan_number
    df = df.sort_values(by=['business_id', 'application_number', 'loan_number'])

    # Initialize a list to store the results
    results = []

    # Group by business_id and apply a custom function to extract features
    for business_id, group in df.groupby('business_id'):
        last_approved_loan_id = None
        for index, row in group.iterrows():
            # Exclude the current loan application from the historical data
            historical_data = group[
                (group['application_number'] < row['application_number']) |
                (group['loan_number'] < row['loan_number'])
            ]

            # Extract features from the historical data
            num_previous_applications = len(historical_data)
            mean_principal_previous = historical_data['principal'].mean()
            mean_owing_previous = historical_data['total_owing_at_issue'].mean()
            num_late_payments_previous = historical_data['paid_late'].sum()

            # Use the current loan application's employee count if no previous data
            if historical_data.empty:
                mean_employee_count_previous = row['employee_count']
            else:
                mean_employee_count_previous = historical_data['employee_count'].mean()

            # Count the number of loan applications in each approval status category
            approval_status_counts = historical_data['approval_status'].value_counts().to_dict()
            approval_status_features = {f'num_{status.lower()}_previous': approval_status_counts.get(status, 0) for status in known_approval_statuses}

            # Find the last approved loan_id
            approved_historical_data = historical_data[historical_data['approval_status'] == 'Approved']
            if not approved_historical_data.empty:
                last_approved_loan_id = approved_historical_data.iloc[-1]['loan_id']

            # Append the results to the list
            features = {
                'loan_id': row['loan_id'],
                'num_previous_applications': num_previous_applications,
                'mean_principal_previous': mean_principal_previous,
                'mean_owing_previous': mean_owing_previous,
                'num_late_payments_previous': num_late_payments_previous,
                'mean_employee_count_previous': mean_employee_count_previous,
                'last_approved_loan_id': int(last_approved_loan_id) if last_approved_loan_id is not None else None,
            }
            features.update(approval_status_features)
            results.append(features)

    # Convert the results list to a DataFrame
    features_df = pd.DataFrame(results)

    return features_df

def calculate_loan_repayment_features(df: pd.DataFrame, transaction_types: List[str]) -> pd.DataFrame:
    """
    Calculate various features related to loan repayment history from a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns ['loan_id', 'paid_at', 'amount', 'transaction_type'].
    transaction_types : List[str]
        List of transaction types to consider (e.g., ['deposit', 'discount']).

    Returns
    -------
    pd.DataFrame
        Dataframe with calculated features for each loan.

    Notes
    -----
    The function assumes that 'paid_at' is in datetime format and 'amount' is a numeric column.
    Missing values in 'transaction_type' are replaced with 'unknown'.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    required_columns = {'loan_id', 'paid_at', 'amount', 'transaction_type'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

    if not isinstance(transaction_types, list) or not all(isinstance(t, str) for t in transaction_types):
        raise TypeError("transaction_types must be a list of strings.")

    # Ensure 'paid_at' is in datetime format
    df['paid_at'] = pd.to_datetime(df['paid_at'])

    # Fill missing transaction_type values with 'unknown'
    df['transaction_type'].fillna('unknown', inplace=True)

    # Group by loan_id and calculate features
    grouped = df.groupby('loan_id').agg(
        total_amount_paid=('amount', 'sum'),
        num_payments=('amount', 'count'),
        max_payment_amount=('amount', 'max'),
        first_payment_date=('paid_at', 'min'),
        last_payment_date=('paid_at', 'max'),
    )

    # Calculate additional features
    grouped['duration'] = (grouped['last_payment_date'] - grouped['first_payment_date']).dt.days
    grouped['single_repayment'] = (grouped['num_payments'] == 1).astype(int)

    # Calculate counts and totals for each transaction type
    for transaction_type in transaction_types:
        grouped[f'num_{transaction_type}'] = df[df['transaction_type'] == transaction_type].groupby('loan_id')['amount'].count()
        grouped[f'total_{transaction_type}_amount'] = df[df['transaction_type'] == transaction_type].groupby('loan_id')['amount'].sum()

    # Fill NaN values with 0 for counts and totals
    grouped.fillna(0, inplace=True)

    # Drop the intermediate first and last payment date columns
    grouped.drop(columns=['first_payment_date', 'last_payment_date'], inplace=True)

    return grouped.reset_index()