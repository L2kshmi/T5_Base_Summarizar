import json
import sys
import jsonify
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import textwrap

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)


from flask import Flask, request, render_template
import requests
app = Flask(__name__)



MODEL_NAME = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

class NewsSummaryModel(pl.LightningModule):
    
  def __init__(self):
    super().__init__()
    self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)

  def forward(self, input_ids, attention_mask, decoder_attention_mask, labels = None):

   output = self.model(
       input_ids,
       attention_mask=attention_mask,
       labels=labels,
       decoder_attention_mask=decoder_attention_mask
   )

   return output.loss, output.logits 

model_saved = 'summarizar.pt'
trained_model = torch.load(model_saved)

def summarize(text):
    text_encoding = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    generated_ids = trained_model.model.generate(
    input_ids=text_encoding["input_ids"],
    attention_mask=text_encoding["attention_mask"],
    max_length=150,
    num_beams=2,
    repetition_penalty=2.5,
    length_penalty=1.0,
    early_stopping=True
    )

    preds = [
    tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    for gen_id in generated_ids
    ]

    return "".join(preds)


# @app.route('/')
# def hello():
#     return 'Hello World!'

@app.route('/', methods=['POST', 'GET'])
def predict():
    summary = "swa"
    text = "juu"
    print(text)
    if request.method == 'POST' :
        text = request.json["actualText"]
        print(text)   
        summary = summarize(text)
        print(summary)
        return {"summary":summary}   

if __name__ == "__main__":
  app.run(host="0.0.0.0")