from sdg_clf.evaluation import get_predictions
from typing import Union

def evaluate(method: str = "sdg_clf", tweet: bool = False, split: str = "test", model_types: Union[str, list[str]] = None,
                    model_weights: Union[str, list[str]] = None, n_samples: int = None, overwrite: bool = False):
    if method == "sdg_clf":
        predictions = []
        for model_type, model_weight in zip(model_types, model_weights):
            prediction = get_predictions(method=method, tweet=tweet, split=split, model_type=model_type, model_weight=model_weight)



# predictions = get_predictions(method="osdg", tweet=False, n_samples=2) # passed
# print(predictions)
# predictions = get_predictions(method="osdg", tweet=True, n_samples=2) # passed
# print(predictions)
predictions = get_predictions(method="aurora", tweet=False, n_samples=2)
print(predictions)
predictions = get_predictions(method="aurora", tweet=True, n_samples=2)
print(predictions)
predictions = get_predictions(method="sdg_clf", tweet=False, model_type="albert-large-v2",
                              model_weight="best_albert.pt", n_samples=2)
print(predictions)
predictions = get_predictions(method="sdg_clf", tweet=False, n_samples=2)
print(predictions)