"""Exploratory data analysis utilities."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Visualization

def plot_histograms(df: pd.DataFrame, n_cols: int = 3, skip_columns: list = None):
    """
    Plots a grid of histograms for each numeric column in a DataFrame using Seaborn,
    skipping columns that have the same value (zero variance) or are specified in the skip list.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot histograms for.
    n_cols : int, optional
        The number of columns in the grid layout (default is 3).
    skip_columns : list, optional
        A list of column names to be excluded from plotting (default is None).

    Returns
    -------
    None
        This function displays the histogram plots for each numeric column.
    """
    if skip_columns is None:
        skip_columns = []

    # Select only numeric columns and exclude any in the skip list
    numeric_columns = [col for col in df.select_dtypes(include='number').columns if col not in skip_columns]

    # Filter out columns with zero variance (i.e., same value)
    numeric_columns = [col for col in numeric_columns if df[col].nunique() > 1]

    # If no columns remain after filtering, exit the function
    if not numeric_columns:
        print("No numeric columns with varying values to plot.")
        return

    # Set the style for seaborn plots
    sns.set(style="whitegrid")

    # Determine the number of rows needed based on the number of numeric columns and the number of columns in the grid
    n_rows = (len(numeric_columns) + n_cols - 1) // n_cols  # Ceiling division to ensure all columns fit

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()  # Flatten in case there is only one row

    # Loop through each numeric column and plot the histogram
    for i, col in enumerate(numeric_columns):
        sns.histplot(df[col], bins=30, kde=True, ax=axes[i])
        axes[i].set_title(f'Histogram for {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_boxplot(df, variables=None, figsize=(12, 8)):
    """
    Plots a boxplot for the specified numeric variables in a DataFrame to visually compare their ranges.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot.
    variables : list, optional
        A list of column names to plot. If None, plots all numeric columns. Default is None.
    figsize : tuple, optional
        The size of the figure for the boxplot. Default is (12, 8).

    Returns
    -------
    None
        Displays a boxplot for the specified numeric variables in the DataFrame.
    """
    # Select only numeric columns if variables are not specified
    if variables is None:
        variables = df.select_dtypes(include='number').columns
    else:
        # Ensure the specified variables are numeric
        variables = [var for var in variables if var in df.select_dtypes(include='number').columns]
    
    # Check if there are any variables to plot
    if not variables:
        print("No numeric variables to plot.")
        return

    # Set figure size
    plt.figure(figsize=figsize)

    # Plot the boxplot for the specified variables
    sns.boxplot(data=df[variables])
    
    # Set plot details
    plt.title('Boxplot of Selected Numeric Variables')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


# Utilities

def print_folder_tree(directory_path: Path, prefix: str = ""):
    """
    Recursively prints the directory tree structure starting from the given directory path.

    Parameters
    ----------
    directory_path : pathlib.Path
        The directory path to print the tree structure for.
    prefix : str, optional
        A string used to format the output, providing indentation for the tree structure.
        Default is an empty string.

    Returns
    -------
    None
        This function prints the directory tree structure to the console.
    """
    # List all items in the given directory
    items = list(directory_path.iterdir())

    # Iterate over each item
    for index, item in enumerate(items):
        # Print the item with a tree structure
        connector = "└── " if index == len(items) - 1 else "├── "
        print(prefix + connector + item.name)

        # If the item is a directory, recursively print its contents
        if item.is_dir():
            # Use the appropriate prefix for sub-items
            extension = "    " if index == len(items) - 1 else "│   "
            print_folder_tree(item, prefix + extension)