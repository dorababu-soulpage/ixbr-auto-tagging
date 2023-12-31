"""
Title: 
    Table Pretrained Model class

Description:
    This file has the Pretrained model class used in training time.

Takeaways:
    - Since this is pytorch_lightning, we need the same class to test/predict.
    - Can modify it to Huggingface structure to future scope.

Author: purnasai@soulpage
Date: 10-10-2023
"""

import torch
import pandas as pd
import lightning.pytorch as pl
from ast import literal_eval

from typing import List, Dict, Tuple
from .utils import System, FileManager
from torchmetrics.classification import F1Score
from torchmetrics import ConfusionMatrix, Precision
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import logging
logger = logging.getLogger(__name__)

SEED= 33
System().set_seed(SEED)
device = System().get_device_to_compute()

CONFIG_PATH = "config.yaml"
yaml_obj = FileManager().load_yaml(CONFIG_PATH)

## list of table tags/labels we used at train time in the proper order
data = FileManager().read_text_file(yaml_obj["LABLES"]["Filepath"])
labels: List[str] = literal_eval(data)

print("Predefined Table labels:", len(labels))
label2id: Dict = {lable: idx for idx, lable in enumerate(labels)}
id2label: Dict = {index: label for label, index in label2id.items()}


class NameMappingModel(pl.LightningModule):
    """Class for ML model realted to Tables,
    this is a pytorch lightning class. this needs
    forward, training_step, validation_step and test_step
    functions.

    This is the exact class used for training as well. same class is
    initiated once again to load trained model."""

    def __init__(self, labels, label2id, id2label):
        super().__init__()
        self.all_test_labels = []
        self.all_test_preds = []
        self.labels = labels
        self.finbert = AutoModelForSequenceClassification.from_pretrained(
            yaml_obj["IXBRL_MODELS"]["Tokenizer"],
            num_labels=len(labels),
            # problem_type="multi_class_classification",
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )
        self.f1 = F1Score(task="multiclass", num_classes=len(labels))
        self.Conf_matrix = ConfusionMatrix(task="multiclass", num_classes=len(labels))
        self.macro_precision = Precision(
            task="multiclass", average="macro", num_classes=len(labels)
        )
        self.micro_precision = Precision(
            task="multiclass", average="micro", num_classes=len(labels)
        )
        # precion-recall curve for multiclass doesn't plot well in the figure, so avoiding it.

    def forward(self, x, y):
        output = self.finbert(**x, labels=y)
        return output

    def training_step(self, batch, batch_idx):
        data, labels = batch
        # outputs = self.finbert(**data, labels = labels)
        outputs = self.forward(x=data, y=labels)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        train_f1_score = self.f1(preds, labels)
        train_macro_precision = self.macro_precision(preds, labels)
        train_micro_precision = self.micro_precision(preds, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_f1", train_f1_score)
        self.log("train_macro_precision", train_macro_precision)
        self.log("train_micro_precision", train_micro_precision)
        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        # outputs = self.finbert(**data, labels = labels)
        outputs = self.forward(x=data, y=labels)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        val_f1_score = self.f1(preds, labels)
        val_macro_precision = self.macro_precision(preds, labels)
        val_micro_precision = self.micro_precision(preds, labels)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_f1", val_f1_score)
        self.log("val_macro_precision", val_macro_precision)
        self.log("val_micro_precision", val_micro_precision)
        return loss

    def test_step(self, batch, batch_idx):
        data, labels = batch
        # outputs = self.finbert(**data, labels = labels)
        outputs = self.forward(x=data, y=labels)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        self.all_test_labels.append(labels)
        self.all_test_preds.append(preds)

        test_f1_score = self.f1(preds, labels)

        test_macro_precision = self.macro_precision(preds, labels)
        test_micro_precision = self.micro_precision(preds, labels)

        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_f1", test_f1_score)
        self.log("test_macro_precision", test_macro_precision)
        self.log("test_micro_precision", test_micro_precision)
        return loss

    def on_test_epoch_end(self):
        labels = torch.cat(self.all_test_labels)
        preds = torch.cat(self.all_test_preds)

        test_final_f1_score = self.f1(preds, labels)
        self.log("test_final_f1_score:", test_final_f1_score)

        conf_mat = self.Conf_matrix(preds, labels)
        computed_confusion = conf_mat.detach().cpu().numpy().astype(int)
        df_cm = pd.DataFrame(computed_confusion, index=self.labels, columns=self.labels)
        df_cm.to_excel("confusion_matrix.xlsx")

    def predict_step(self, batch, batch_idx):
        # this is acting same as test_step in our case
        # we dont need this anymore
        data, label = batch
        outputs = self.forward(x=data, y=label)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)
        result = [id2label[pred_label.item()] for pred_label in predicted_labels]
        return result

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-5)


modeleval = NameMappingModel.load_from_checkpoint(
    checkpoint_path="Models1/Table_Inline_Model/SECtag_RarelabelModel-epoch=39-val_loss=0.26.ckpt",
    labels=labels,
    label2id=label2id,
    id2label=id2label,
    map_location=device,
    strict=False,
)

# disable dropout, etc... with eval mode
modeleval = modeleval.eval()
modeleval = modeleval.to(device)
tokenizer = AutoTokenizer.from_pretrained(
    "soleimanian/financial-roberta-large-sentiment"
)


def predict_table_tags(data) -> Tuple[List, List]:
    """function to predict table tags"""
    
    logger.info("2.4. Predicting table tags......")
    texts = []
    predicted_labels = []
    
    # predict with the model
    with torch.no_grad():
        for table_data in data:
            for row in table_data:
                text, tag = row.split("==")
                inputs = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=32,
                    return_tensors="pt",
                )

                # random tag just to pass to model to match with syntax
                tag = "us-gaap:StockholdersEquity"
                y = label2id[tag]
                y = torch.tensor(y).squeeze()
                y = y.to(device)

                for key in inputs:
                    inputs[key] = inputs[key].to(device)

                outputs = modeleval(inputs, y)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

                # truelabels = [id2label[label.item()] for label in labels]
                predlabels = [id2label[label.item()] for label in preds][0]

                texts.append(text)
                predicted_labels.append(predlabels)

    return texts, predicted_labels
