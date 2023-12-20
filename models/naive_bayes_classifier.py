import numpy as np
import pandas as pd


def calculate_class_probability(
    df: pd.DataFrame, feature_cols: list[str], target_col: str, values: list[int]
) -> dict[int, float]:
    """
    Calculate class probabilities for Naïve Bayes Classifier.

    Parameters:
    - df: DataFrame, input data
    - feature_cols: list of str, feature columns
    - target_col: str, target column
    - values: list of int, unique values for features

    Returns:
    - class_map: dict, class probabilities
    """

    class_probabilities = {}

    for cls in df[target_col].unique():
        class_df = df[df[target_col] == cls]

        feature_probabilities = [
            class_df[class_df[col] == val][target_col].count() / len(class_df)
            for val, col in zip(values, feature_cols)
        ]

        class_probabilities[cls] = np.prod(
            feature_probabilities + [len(class_df) / len(df)]
        )

    return class_probabilities


def calculate_normalized_probabilities(
    class_probabilities: dict[int, float]
) -> dict[int, float]:
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


def naive_bayes_classifier(
    df: pd.DataFrame, feature_cols: list[str], target_col: str, values: list[int]
) -> tuple[pd.DataFrame, int]:
    """
    Naïve Bayes Classifier.

    Parameters:
    - df: DataFrame, input data
    - feature_cols: list of str, feature columns
    - target_col: str, target column
    - values: list of int, unique values for features

    Returns:
    - result_df: DataFrame, result containing class and probability information
    - max_prob_class: int, the class with the highest probability
    """

    class_probabilities = calculate_class_probability(
        df, feature_cols, target_col, values
    )
    normalized_probabilities = calculate_normalized_probabilities(class_probabilities)

    result_df = create_result_dataframe(class_probabilities, normalized_probabilities)
    max_prob_class = result_df["Probability"].idxmax()

    return result_df, max_prob_class


def main() -> None:
    pass


if __name__ == "__main__":
    main()
