import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import textwrap

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)
from tqdm.auto import tqdm


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

text = """An old man lived in the village. He was one of the most unfortunate people in the world. 
The whole village was tired of him; he was always gloomy, he constantly complained and was always in a bad mood. 
The longer he lived, the more bile he was becoming and the more poisonous were his words. People avoided him, 
because his misfortune became contagious. It was even unnatural and insulting to be happy next to him. 
He created the feeling of unhappiness in others. But one day, when he turned eighty years old, 
an incredible thing happened. Instantly everyone started hearing the rumour: “An Old Man is happy today, 
he doesn’t complain about anything, smiles, and even his face is freshened up.” The whole village gathered together. 
The old man was asked: What happened to you? 
“Nothing special. Eighty years I’ve been chasing happiness, and it was useless. 
And then I decided to live without happiness and just enjoy life. That’s why I’m happy now.” – An Old Man """

print(summarize(text))