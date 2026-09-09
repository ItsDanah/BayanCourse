"""Lab 3 starter: dataset construction and split integrity."""

import pandas as pd

from datasets import Dataset, DatasetDict
from sklearn.model_selection import GroupShuffleSplit


def build_topic_dataset(
    path="data/raw/bayan_feedback.csv",
    group_col="citizen_group_id",
):
    # Load the supplied feedback dataset
    df = pd.read_csv(path)

    # Split citizens into train and temporary validation/test groups
    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=42,
    )

    train_idx, temp_idx = next(
        first_split.split(
            df,
            groups=df[group_col],
        )
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    # Split the temporary group equally into validation and test
    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=42,
    )

    validation_idx, test_idx = next(
        second_split.split(
            temp_df,
            groups=temp_df[group_col],
        )
    )

    validation_df = temp_df.iloc[validation_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    # Convert pandas DataFrames to Hugging Face datasets
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(
                train_df,
                preserve_index=False,
            ),
            "validation": Dataset.from_pandas(
                validation_df,
                preserve_index=False,
            ),
            "test": Dataset.from_pandas(
                test_df,
                preserve_index=False,
            ),
        }
    )

    return dataset