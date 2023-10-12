"""
Title: 
    Model Class

Description:
    This file has all models(cover, notes) packed in single class to use in app.py file 

Takeaways:
    - Same tokenizer used for Cover and Notes page.
    - Table model is in another file.
    - Cover page max_length=64, Notes max_length=128

Author: purnasai@soulpage
Date: 10-10-2023
"""

import torch
import warnings

from .utils import post_process
from transformers import AutoTokenizer, AutoModelForTokenClassification

warnings.filterwarnings("ignore")

class Xbrl_Tag():
    """ML Model class with tokenizers and Models loaded.
    These models are already trained on the 10-Q dataset of 
    50 to 70 companies.
    """
    def __init__(self):
        self.device        =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dei_tokenizer =  AutoTokenizer.from_pretrained("soleimanian/financial-roberta-large-sentiment",
                                                    add_prefix_space= True,
                                                    do_lower_case = True
                                                    )

        self.dei_model = AutoModelForTokenClassification.from_pretrained(r"Models/DEI_Model")
        self.dei_model = self.dei_model.to(self.device)
        
        self.notes_model = AutoModelForTokenClassification.from_pretrained(r"Models/Notes_Model")
        self.notes_model = self.notes_model.to(self.device)


    def predict_dei_tags(self, total_rows):
        """Function to predict DEI/Cover page entities
        and returns reconstructed input sentence with 
        output  tags where each tag is output of each input word
        """
        original_inputs, total_inputs, total_outputs = [],[],[]
        for input_row in total_rows:
            joined_text = " ".join(input_row)

            new_inputs = self.dei_tokenizer(joined_text,
                                            padding='max_length',
                                            truncation=True,
                                            max_length= 64,
                                            return_tensors='pt',
                                            is_split_into_words= False,
                                            )
            new_inputs = {key:val.to(self.device) for key,val in new_inputs.items()}

            with torch.no_grad():
                # predict
                new_logits = self.dei_model(**new_inputs).logits

            new_predictions = torch.argmax(new_logits, dim=2)
            new_predicted_token_labels = [self.dei_model.config.id2label[t.item()] for t in new_predictions[0]]
            decoded_string = self.dei_tokenizer.convert_ids_to_tokens(new_inputs["input_ids"][0])
            reconstructed_row, reconstructed_predictions = post_process(decoded_string, new_predicted_token_labels)
            
            original_inputs.append(joined_text)
            total_inputs.append(reconstructed_row)
            total_outputs.append(reconstructed_predictions)

        return original_inputs, total_inputs, total_outputs
    
    def predict_notes_tags(self, total_rows):
        """Function to predict tags in the Notes section in Filings.
        """
        total_inputs, total_outputs = [],[]
        for input_row in total_rows:
            joined_text = " ".join(input_row)

            new_inputs = self.dei_tokenizer(joined_text,
                                            padding='max_length',
                                            truncation=True,
                                            max_length=128,
                                            return_tensors='pt',
                                            is_split_into_words= False,
                                            )

            new_inputs = {k:v.to(self.device) for k,v in new_inputs.items()}

            with torch.no_grad():
                new_logits = self.notes_model(**new_inputs).logits

            new_predictions = torch.argmax(new_logits, dim=2)
            new_predicted_token_class = [self.notes_model.config.id2label[t.item()] for t in new_predictions[0]]
            decoded_string = self.dei_tokenizer.convert_ids_to_tokens(new_inputs["input_ids"][0])

            reconstructed_sentence = ""
            reconstructed_labels = []

            for subtoken,label in zip(decoded_string, new_predicted_token_class):
                # Skip special tokens and subtoken markers
                if subtoken != "<pad>":
                    if subtoken in ["<s>", "</s>"]  or subtoken.startswith("Ġ"):
                        reconstructed_sentence =  reconstructed_sentence+" "+subtoken.replace("Ġ", "")
                        reconstructed_labels.append(label)
                    else:
                        reconstructed_sentence += subtoken
            
            total_inputs.append(reconstructed_sentence)
            total_outputs.append(reconstructed_labels)

        return total_inputs, total_outputs