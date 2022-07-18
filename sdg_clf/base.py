import torch
import transformers
from sdg_clf import utils


# transformer class with a model and a tokenizer
class Transformer:
    def __init__(self, model: torch.nn.Module, tokenizer: transformers.PreTrainedTokenizer, max_length: int = 260):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = model.device

    def prepare_input_ids(self, text: str) -> torch.Tensor:
        input_ids = self.tokenizer(text, add_special_tokens=False)
        input_ids = input_ids["input_ids"]
        input_ids = [input_ids[x:x + self.max_length - 2] for x in range(0, len(input_ids), self.max_length)]
        # add bos and eos tokens to each input_ids
        input_ids = [[self.tokenizer.cls_token_id] + x + [self.tokenizer.eos_token_id] for x in input_ids]
        # pad input_ids to max_length
        input_ids[-1] = input_ids[-1] + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids[-1]))
        return torch.tensor(input_ids)

    def prepare_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        rows, columns = input_ids.shape
        attention_mask = torch.ones((rows-1, columns))
        last_mask = torch.tensor([[1 if id != self.tokenizer.pad_token_id else 0 for id in input_ids[-1, :]]])
        attention_mask = torch.concat((attention_mask, last_mask), dim=0)
        return attention_mask

    def prepare_model_inputs(self, text: str) -> dict[str, torch.Tensor]:
        input_ids = self.prepare_input_ids(text)
        attention_mask = self.prepare_attention_mask(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def predict_sample_no_threshold(self, text: str) -> torch.Tensor:
        """
        Predict the sample with the model and return the probabilities of each class.

        The text is split into chunks of 260 tokens.
        There can therefore be several chunks.
        A prediction is made on each chunk and the sigmoid activation function applied to each.
        The output is of shape (n_chunks, 17)

        Args:
            text: the text to predict on

        Returns:
            predictions for each chunk

        """
        # get model inputs
        model_inputs = self.prepare_model_inputs(text)
        # get predictions
        with torch.no_grad():
            # set inputs to device
            model_inputs = utils.move_to(model_inputs, self.device)
            # forward pass
            outputs = self.model(**model_inputs).logits
            outputs = torch.sigmoid(outputs).cpu()
        return outputs


