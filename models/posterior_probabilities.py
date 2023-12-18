import pandas as pd


def rename_index(df: pd.DataFrame) -> None:
    """
    Rename the index of the DataFrame according to the specified pattern.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    """

    df.columns = [f"h{i}" for i in range(1, len(df.columns) + 1)]
    df.index = ["P(Hi)"] + [f"P(E{i}/Hi)" for i in range(1, len(df))]


def fill_nan(row: pd.Series) -> pd.Series:
    """
    Fill NaN values in a DataFrame row with the complementary probability.

    Parameters:
    - row (pd.Series): The input row of the DataFrame.

    Returns:
    - pd.Series: The row with NaN values filled.
    """

    nan_indices = row.index[row.isna()]

    if len(nan_indices) > 0:
        row_sum = 1 - row.sum()
        row[nan_indices] = row_sum

    return row


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the DataFrame by renaming the index and filling NaN values.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: Processed DataFrame.
    """
    rename_index(df)
    return df.apply(fill_nan, axis=1)


def calc_posterior_probabilities(
    df: pd.DataFrame, evidences: list[str], hypothesis: list[str]
) -> pd.Series:
    """
    Calculate posterior probabilities based on the specified evidences and hypothesis of a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - evidences (list): List of evidences indices.
    - hypothesis (list): List of hypothesis indices.

    Returns:
    - pd.Series: Resulting posterior probabilities.
    """

    evidences = ["P(Hi)"] + evidences
    return df.loc[evidences, hypothesis].prod() / (df.loc[evidences].prod()).sum()


def posterior_probabilities(
    df: pd.DataFrame, evidences: list[str], hypothesis: list[str]
) -> (pd.DataFrame, str):
    """
    Calculate posterior probabilities and identify the maximum evidence.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - evidences (list): List of evidences indices.
    - hypothesis (list): List of hypothesis indices.

    Returns:
    - pd.DataFrame: Resulting posterior probabilities DataFrame.
    - str: Index of the maximum posterior probability.
    """

    result = calc_posterior_probabilities(df, evidences, hypothesis)

    result_df = pd.DataFrame(result).T
    result_df.index = ["Posterior Probabilities"]

    max_posterior_probability = result_df.idxmax(axis=1).iloc[0]

    return result_df, max_posterior_probability


def main() -> None:
    pass


if __name__ == "__main__":
    main()
