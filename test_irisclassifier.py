from typing import Text
import irisclassifier
import pytest

performance_thresholds = (0.50, 0.75, 0.95)

@pytest.mark.parametrize('th', performance_thresholds)

def test_evaluation(th):

    i = irisclassifier.IrisClassifier()
    i.ingestion()
    i.segregation()
    i.train()
    res = i.evaluation()

    assert res > th
