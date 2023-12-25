import numpy as np
import pandas as pd

import math


def exponential_pdf(x: float, mean: float) -> float:
    """
    Calculate the Exponential Probability Density Function (PDF).

    Parameters:
    - x: The input value.
    - mean: The mean (average) of the distribution.

    Returns:
    - pdf: float, The probability density function value for the given input in the Exponential distribution.
    """

    rate_parameter = 1 / mean
    pdf = rate_parameter * math.exp(-rate_parameter * x)
    return pdf


def exponential_pdfs(
    df: pd.DataFrame, feature_col: str, target_col: str, x: float
) -> list[float]:
    """
    Calculate Exponential Probability Density Functions for each class.

    Parameters:
    - df: DataFrame, input data
    - feature_col: str, the feature column
    - target_col: str, the target column
    - x: float, the input value

    Returns:
    - pdfs: list of float, Exponential PDF values for each class.
    """

    return [
        exponential_pdf(x, class_df[feature_col].mean())
        for _, class_df in df.groupby(target_col)
    ]


def calc_class_prob(df: pd.DataFrame, target_col: str) -> list[float]:
    """
    Calculate class probabilities.

    Parameters:
    - df: DataFrame, input data
    - target_col: str, the target column

    Returns:
    - prob: list of float, class probabilities.
    """

    return [len(class_df) / len(df) for _, class_df in df.groupby(target_col)]


def calculate_class_probability(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    values: list[float],
) -> np.ndarray:
    """
    Calculate class probabilities for the Mix Naïve Bayes Classifier.

    Parameters:
    - df: DataFrame, input data
    - feature_cols: list of str, Feature columns
    - target_col: str, the target column
    - values: list of float, input values for each feature

    Returns:
    - class_probabilities: np.ndarray, class probabilities.
    """

    individual_probabilities = [
        exponential_pdfs(df, col, target_col, value)
        for value, col in zip(values, feature_cols)
    ] + [calc_class_prob(df, target_col)]

    class_probabilities = dict(
        zip(df[target_col].unique(), np.prod(individual_probabilities, axis=0))
    )

    return class_probabilities


def calculate_normalized_probabilities(
    class_probabilities: dict[str, float]
) -> dict[str, float]:
    """
    Calculate normalized class probabilities.

    Parameters:
    - class_probabilities: dict, class probabilities

    Returns:
    - normalized_probabilities: dict, normalized class probabilities
    """

    total_probability = sum(class_probabilities.values())
    normalized_probabilities = {
        cls: prob / total_probability for cls, prob in class_probabilities.items()
    }

    return normalized_probabilities


def create_result_dataframe(
    class_probabilities: dict[int, float], normalized_probabilities: dict[int, float]
) -> pd.DataFrame:
    """
    Create a DataFrame containing class probabilities and normalized probabilities.

    Parameters:
    - class_probabilities: dict, class probabilities
    - normalized_probabilities: dict, normalized class probabilities

    Returns:
    - result_df: DataFrame, result containing class and probability information
    """

    class_df = pd.DataFrame(list(class_probabilities.items()), columns=["Class", "Map"])
    normalized_df = pd.DataFrame(
        list(normalized_probabilities.items()),
        columns=["Class", "Probability"],
    )

    return pd.merge(class_df, normalized_df, on="Class").set_index("Class")


def exponential_naive_bayes_classifier(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    values: list[float],
) -> tuple[pd.DataFrame, int]:
    """
    Exponential Naïve Bayes Classifier.

    Parameters:
    - df: DataFrame, input data
    - feature_cols: List of str, Feature columns
    - target_col: str, the target column
    - values: List of float, input values for each feature

    Returns:
    - result_df: DataFrame, result containing class and probability information
    - max_prob_class: int, the class with the highest probability
    """

    class_probabilities = calculate_class_probability(
        df, feature_cols, target_col, values
    )
    normalized_probabilities = calculate_normalized_probabilities(class_probabilities)

    result_df = create_result_dataframe(class_probabilities, normalized_probabilities)
    result_df["Map"] = result_df["Map"].apply(lambda x: f"{x:.2e}")

    max_prob_class = result_df["Probability"].idxmax()

    return result_df, max_prob_class


def main() -> None:
    pass


if __name__ == "__main__":
    main()
