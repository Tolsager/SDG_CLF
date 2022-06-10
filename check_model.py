import torch
import transformers


def check_model(path_model: str, model_type):
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=17)
    model.cuda()
    model.load_state_dict(torch.load(path_model))
    input_ids = torch.randint(200, (1, 260))
    input_ids = input_ids.to("cuda")
    out = model(input_ids)
    return out


if __name__ == "__main__":
    out = check_model("noble-resonance-40.pt", "albert-large-v2")
    print(out)
