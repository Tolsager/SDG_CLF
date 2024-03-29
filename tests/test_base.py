import torch
import transformers

from sdg_clf import base


class TestTransformer:
    tokenizer = transformers.AutoTokenizer.from_pretrained("albert-base-v2")
    model = transformers.AutoModelForSequenceClassification.from_pretrained("albert-base-v2", num_labels=17)
    transformer = base.Transformer(model, tokenizer)
    text = "This is a test"

    def test_prepare_input_ids(self):
        input_ids = self.transformer.prepare_input_ids(self.text)
        assert isinstance(input_ids, torch.Tensor)
        assert input_ids.dim() == 2

    def test_prepare_attention_mask(self):
        input_ids = torch.ones((1, 260))
        input_ids[0, 10:] = self.tokenizer.pad_token_id
        attention_mask = self.transformer.prepare_attention_mask(input_ids)
        assert isinstance(attention_mask, torch.Tensor)
        assert attention_mask.dim() == 2
        assert attention_mask.shape == (1, 260)
        assert torch.all(attention_mask[0, :10] == torch.ones(10))
        assert torch.all(attention_mask[0, 10:] == torch.zeros(250))

    def test_prepare_model_inputs(self):
        model_inputs = self.transformer.prepare_model_inputs(self.text)
        assert isinstance(model_inputs, dict)
        assert "input_ids" in model_inputs
        assert "attention_mask" in model_inputs
        assert model_inputs["input_ids"].dim() == 2
        assert model_inputs["attention_mask"].dim() == 2

    def test_predict_sample_no_threshold(self):
        predictions = self.transformer.predict_sample_no_threshold(self.text)
        assert isinstance(predictions, torch.Tensor)
        assert predictions.shape == (1, 17)
        assert torch.all(predictions >= 0) and torch.all(predictions <= 1)

    def test_predict_multiple_samples_no_threshold(self):
        predictions = self.transformer.predict_multiple_samples_no_threshold([self.text, self.text])
        assert isinstance(predictions, list)
        assert len(predictions) == 2
        assert predictions[0].shape == (1, 17)
        assert predictions[1].shape == (1, 17)
