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

import os
import nltk

from bs4 import BeautifulSoup, Comment
from nltk.tokenize import sent_tokenize
from .utils import clean_text

nltk.download('punkt')

import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger(__name__)


def get_text_outside_table(content_between_comments):
    """THis function takes the part of a page out of many pages,
    removes table from the page, returns remaining text."""
    soup1 = BeautifulSoup(content_between_comments, "lxml")
    # Remove all the tables from the HTML content
    for table in soup1.find_all('table'):
        table.extract()

    # Remove all the elements with table-related tags from the HTML content
    for tag in ['tbody', 'thead', 'tfoot', 'tr', 'th', 'td']:
        for element in soup1.find_all(tag):
            element.extract()

    # Get the text outside the tables and table-related elements
    text_outside_tables = soup1.get_text()
    return text_outside_tables, soup1


def process_text(paragraph):
    """This splits paragraph into multiple sentences.
    these sentences are cleaned to remove $ and others
    """
    ip_text = []

    sentences = sent_tokenize(paragraph)
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

def get_NER_Data(html_data):
    """Takes html as input, finds html code of text outside tabels,
    finds P/span tags, extract, cleans, splits the text."""
    logger.info("3.1. Started collecting entire text, not just pages with notes heading..")
    total_ip_texts = []
    # this eliminates tables in Notes section
    text_content, html_content = get_text_outside_table(html_data)

    # if Paragraph tags found
    if html_content.find_all("p"):
        for p in html_content.find_all("p"):
            text = p.get_text()

            ip_text = process_text(text)
            if ip_text:
                ip_text = " ".join(ip_text)
                ip_text = clean_text(ip_text)
                total_ip_texts.append(ip_text.split(" "))

    else: # else if span tags found
        for span in html_content.find_all("span"):
            text = span.get_text()

            ip_text = process_text(text)
            if ip_text:
                ip_text = " ".join(ip_text)
                ip_text = clean_text(ip_text)
                total_ip_texts.append(ip_text.split(" "))

    return total_ip_texts


# def save_rows_and_tags(save_path, folder_name, input_data, output_data):
#     # create folder if not exists

#     save_folder = os.path.join(save_path, folder_name)
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)

#     input_filename = os.path.join(save_folder, "inputs.txt")
#     output_filename = os.path.join(save_folder, "outputs.txt")

#     with open(input_filename, 'w') as f:
#         for row in input_data:
#             f.write("%s\n" % row)

#     with open(output_filename, 'w') as f:
#         for row1 in output_data:
#             f.write("%s\n" % row1)


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