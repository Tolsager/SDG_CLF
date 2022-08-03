![SDG-CLF](https://user-images.githubusercontent.com/73647490/172805470-ffb6a817-7334-40c0-9249-97b8036219ec.jpg)

# Classification of UN Sustainable Development Goals using Tranformers (WORK IN PROGRESS)
Predicts the presence of the United Nations Sustainable Development Goals (SDGs) in texts.

We provide 3 fine-tuned models which can be used as an ensemble.

# Setup
1. Clone the repository and navigate into it
```bash
git clone https://github.com/Tolsager/SDG_CLF/
cd SDG_CLF
```

2. Download everything from ![Google Drive](https://drive.google.com/drive/u/2/folders/1kRPZtGJyI9dRq59wQSMxEeDHgQ-erw_m) and place them in `SDG_CLF/finetuned_models`.
You should then have the following file architecture
```
SDG_CLF
--finetuned_models
  --microsoft
    --deberta-v3-large_model0.pt
  --albert-large-v2_model0.pt
  --roberta-large_model0.pt
```

3. Download the necessary libraries
```bash
pip install -r requirements.txt
```
# Predict on text
4. Run the predict script
Predict with the fine-tuned roberta-large model
```bash
python predict.py --text "Access to safe water, sanitation and hygiene is the most basic human need for health and well-being." --model_weights roberta-large_model0.pt 
```

Predict with the fine-tuned roberta-large model at a costum threshold of 0.3
```bash
python predict.py --text "Access to safe water, sanitation and hygiene is the most basic human need for health and well-being." --model_weights roberta-large_model0.pt --threshold 0.3
```

Predict with an ensemble of the 3 fine-tuned models
```bash
python predict.py --text "Access to safe water, sanitation and hygiene is the most basic human need for health and well-being." --model_weights roberta-large_model0.pt albert-large-v2_model0.pt microsoft/deberta-v3-large_model0.pt 
```

# Predict on CSV file
Predict with roberta-large on file "sample.csv" with a column "text" that the model should predict on. Save the predictions in a csv "sample_predictions.csv"
``bash
python predict.py --file sample.csv --model_weights roberta-large_model0.pt --column text --save_path predictions.csv
```
