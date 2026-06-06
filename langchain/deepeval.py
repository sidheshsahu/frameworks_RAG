import deepeval
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

def test_llm():
    test_case = LLMTestCase(input="...", actual_output="...")
    answer_relevancy_metric = AnswerRelevancyMetric()
    assert_test(test_case, [answer_relevancy_metric])

@deepeval.log_hyperparameters(model="gpt-4.1", prompt_template="...")
def hyperparameters():
    return {
        "temperature": 1,
        "chunk size": 500
    }