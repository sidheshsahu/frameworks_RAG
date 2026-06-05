from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import GeminiModel



model = GeminiModel("gemini-2.5-flash")
task_completion_metric = AnswerRelevancyMetric(model=model)


