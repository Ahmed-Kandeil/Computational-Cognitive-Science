import numpy as np
import pandas as pd


from nltk.tokenize import word_tokenize


def get_unique_words(
    df: pd.DataFrame,
    col: str,
    tokenize: bool = False,
):
    """
    Get unique words from a DataFrame column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        col (str): The column name containing text data.
        tokenize (bool): If True, tokenize the words using nltk. Default is False.

    Returns:
        set: A set of unique words.
    """

    words = df[col].str.cat(sep=" ").lower()
    words = word_tokenize(words) if tokenize else words.split()

    unique_words = set(words)
    return unique_words


def get_words(
    df: pd.DataFrame,
    col: str,
    tokenize: bool = False,
):
    """
    Get words from a DataFrame column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        col (str): The column name containing text data.
        tokenize (bool): If True, tokenize the words using nltk. Default is False.

    Returns:
        list: A list of words.
    """

    words = df[col].str.cat(sep=" ").lower()
    words = word_tokenize(words) if tokenize else words.split()

    return words


def calculate_word_prob(words: list[str], total_words: int, value: str):
    """
    Calculate the probability of a word.

    Parameters:
        words (list): List of words.
        total_words (int): Total number of unique words.
        value (str): The word for which probability is calculated.

    Returns:
        float: Word probability.
    """

    return (words.count(value.lower()) + 1) / (len(words) + total_words)


def calculate_words_prob(words: list[str], total_words: int, values: list[str]):
    """
    Calculate the product of word probabilities.

    Parameters:
        words (list): List of words.
        total_words (int): Total number of unique words.
        values (list): List of words for which probabilities are calculated.

    Returns:
        float: Product of word probabilities.
    """

    return np.prod(
        [calculate_word_prob(words, total_words, value) for value in values.split()]
    )


def calculate_class_probability(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    values: str,
    tokenize: bool = False,
):
    """
    Calculate class probabilities.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        feature_col (str): The column name containing text data.
        target_col (str): The column name containing class labels.
        values (str): Text values for which probabilities are calculated.
        tokenize (bool): If True, tokenize the words using nltk. Default is False.

    Returns:
        dict: Dictionary with class probabilities.
    """

    total_words = len(get_unique_words(df, feature_col))
    class_probabilities = {}

    for cls in df[target_col].unique():
        class_df = df[df[target_col] == cls]
        class_words = get_words(class_df, feature_col, tokenize)
        class_prob = len(class_df) / len(df)

        class_probabilities[cls] = (
            calculate_words_prob(class_words, total_words, values) * class_prob
        )

    return class_probabilities


def calculate_normalized_probabilities(
    class_probabilities: dict[str, float]
) -> dict[str, float]:
    """
    Calculate normalized class probabilities.

    Parameters:
        class_probabilities (dict): Dictionary with class probabilities.

    Returns:
        dict: Dictionary with normalized class probabilities.
    """

    total_probability = sum(class_probabilities.values())
    normalized_probabilities = {
        cls: prob / total_probability for cls, prob in class_probabilities.items()
    }

    return normalized_probabilities


def create_result_dataframe(
    class_probabilities: dict[str, float], normalized_probabilities: dict[str, float]
) -> pd.DataFrame:
    """
    Create a DataFrame with class probabilities.

    Parameters:
        class_probabilities (dict): Dictionary with class probabilities.
        normalized_probabilities (dict): Dictionary with normalized class probabilities.

    Returns:
        pd.DataFrame: Resulting DataFrame.
    """

    class_df = pd.DataFrame(list(class_probabilities.items()), columns=["Class", "Map"])
    normalized_df = pd.DataFrame(
        list(normalized_probabilities.items()),
        columns=["Class", "Probability"],
    )

    return pd.merge(class_df, normalized_df, on="Class").set_index("Class")


def nlp(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    values: str,
    tokenize: bool = False,
) -> tuple[pd.DataFrame, str]:
    """
    Perform NLP analysis and return results.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        feature_col (str): The column name containing text data.
        target_col (str): The column name containing class labels.
        values (str): Text values for which probabilities are calculated.
        tokenize (bool): If True, tokenize the words using nltk. Default is False.

    Returns:
        tuple: Resulting DataFrame and the class with maximum probability.
    """

    class_probabilities = calculate_class_probability(
        df, feature_col, target_col, values, tokenize
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
