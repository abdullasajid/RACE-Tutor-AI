import pandas as pd
import pytest


@pytest.fixture
def sample_race_df():
    return pd.DataFrame(
        [
            {
                "id": "sample-1",
                "article": "Ali studied hard. He passed the exam.",
                "question": "Why did Ali pass?",
                "A": "He slept",
                "B": "He studied hard",
                "C": "He missed school",
                "D": "He forgot the exam",
                "answer": "B",
            }
        ]
    )
