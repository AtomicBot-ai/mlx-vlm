import pytest

from mlx_vlm.evals.math_vista import evaluate_answer, normalize_answer

INTEGER_PROBLEM = {"question_type": "free_form", "answer_type": "integer"}


@pytest.mark.parametrize(
    "response, expected",
    [
        ("I count 7 birds in the picture. However, two are partly hidden.", "7"),
        ("There are 12 people, standing in a row,", "12"),
        ("The population grew to 7,518 people, then fell.", "7518"),
    ],
)
def test_math_vista_integer_ignores_commas_in_prose(response, expected):
    prediction = normalize_answer(response, INTEGER_PROBLEM)

    assert prediction == expected
    assert evaluate_answer(prediction, expected)


def test_math_vista_explicit_integer_answer_takes_precedence_over_last_number():
    response = "The answer is 1,024 after doubling 512."

    assert normalize_answer(response, INTEGER_PROBLEM) == "1024"
