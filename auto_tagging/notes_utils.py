"""
Title: 
    Notes Utilities

Description:
    This file has all the functionalities/utilities required for Notes processing.

Takeaways:
    - we are not processin the tables in the Notes sections.
    - 10-Q has 11 to 12 Notes sections, we can modularize (or) add more context \
    - before each line of text to better result, like we did it in table.

Author: purnasai@soulpage
Date: 10-10-2023
"""
import bs4
import nltk
import logging
import warnings

from typing import List, Tuple
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from .utils import ProcessText, get_text_outside_table

nltk.download('punkt')
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

processtext = ProcessText()


def process_text(paragraph: str) -> List:
    """This splits paragraph into multiple sentences.
    these sentences are cleaned to remove $ and others
    """

    ip_text = []
    sentences: List[str] = sent_tokenize(paragraph)
    
    for sentence in sentences:
        sentence_ip_text = []
        
        for token in sentence.split(" "):
            strip_token = token.replace("$","")
            if strip_token.endswith("."):
                strip_token = strip_token.replace(".","")
            sentence_ip_text.append(strip_token)

        if len(list(set(sentence_ip_text))) > 1:
            ip_text.extend(sentence_ip_text)

    return ip_text

def get_NER_Data(html_data: str) -> List[List[str]]:
    """Takes html as input, finds html code of text outside tabels,
    finds P/span tags, extract, cleans, splits the text."""
    
    logger.info("3.1. Started collecting entire text, not just pages with notes heading..")
    total_ip_texts = []

    # this eliminates tables in Notes section
    text_content, html_content = get_text_outside_table(html_data)

    # if Paragraph tags found
    if html_content.find_all("p"):
        for p in html_content.find_all("p"):
            text: str = p.get_text()

            ip_text = process_text(text)
            if ip_text:
                ip_text = " ".join(ip_text)
                ip_text = processtext.clean_text(ip_text)
                total_ip_texts.append(ip_text.split(" "))

    else: # else if span tags found
        for span in html_content.find_all("span"):
            text: str = span.get_text()

            ip_text = process_text(text)
            if ip_text:
                ip_text = " ".join(ip_text)
                ip_text = processtext.clean_text(ip_text)
                total_ip_texts.append(ip_text.split(" "))

    return total_ip_texts

def clean_notes_outputs(inputs, outputs):
    """Function to filter out sentences with 
    only "O" label entirely"""
    new_inputs = []
    new_outputs = []

    for index in range(len(inputs)):
        label = list(set(outputs[index]))
        if label[0] != "O":
            new_inputs.append(inputs[index])
            new_outputs.append(outputs[index])

    return new_inputs, new_outputs