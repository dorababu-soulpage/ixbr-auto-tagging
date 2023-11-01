"""
Title: 
    General Utils

Description:
    All the General util functions, file processing, html page splitting functions here.

Takeaways:
    - currently as of date, search and replace technique is used to overwrite the html file.
    - Need to work on alternative approach to replace the values.

Author: purnasai@soulpage
Date: 10-10-2023
"""


import re
import yaml
import torch
import random
import warnings
import numpy as np

from ast import literal_eval
from bs4 import BeautifulSoup, Comment


warnings.filterwarnings("ignore")

with warnings.catch_warnings():
    # this is to only avoid deprecation warning in clean_text package
    # but not to all other packages.
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from cleantext import clean


class System:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_device_to_compute(
        self,
    ):
        return self.device

    def set_seed(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class FileManager:
    def __init__(self) -> None:
        pass

    def load_yaml(self, file_path):
        """function to safe load config yaml file"""
        with open(file_path, "r") as file:
            yaml_obj = yaml.safe_load(file)
        return yaml_obj

    def read_text_file(self, file_path):
        """furnction to read plain text file and return data"""
        with open(file_path, "r") as fp:
            data = fp.read()
        return data

    def read_text_file_lines(self, text_file_path):
        """reads text file & converts each row to list"""
        with open(text_file_path, "r") as fp:
            text = fp.readlines()
        text = [literal_eval(text_row) for text_row in text]
        return text

    def read_html_file(self, html_path):
        """fucntion to read html file"""
        with open(html_path, "r", encoding="unicode-escape") as file:
            html_data = file.read()
        return html_data

    def save_html_file(self, html_file_path, html_table_script):
        """save html script of STRING to html page"""
        with open(html_file_path, "w", encoding="utf-8") as f:
            f.write(str(html_table_script))


class HtmlContent:
    def __init__(self) -> None:
        pass

    def extract_until_comments(self, start_comment):
        """function to get html code until the page break"""
        content = ""
        curr = start_comment.find_previous_sibling()
        while curr:
            content = str(curr) + content
            if curr.name == "comment":
                break
            curr = curr.find_previous_sibling()
        return content

    def extract_between_comments(self, start_comment, end_comment):
        """fucntion to get html code from one page start to same page end"""
        content = ""
        curr = start_comment.next_sibling
        while curr and curr != end_comment:
            content += str(curr)
            curr = curr.next_sibling
        return content

    def split_page(self, html_path):
        """this is to split the input html that user uploads in to platform.
        the same html is overwrittern with output tags. Simple split is used,
        rather than the complex split we did in fucntions below
        - split_page_and_extract_text
        - collect_tokens
        in dei_utils.py file"""
        html_data = FileManager().read_html_file(html_path)
        soup = BeautifulSoup(html_data, "lxml")

        # Find all <!-- Field: Page; Sequence> tags
        comments = soup.find_all(
            string=lambda text: isinstance(text, Comment) and "Field: Page;" in text
        )

        # split html page by comments
        if comments:
            # print("Comments found")
            for i in range(len(comments[:1])):  # all pages, not slicing
                start_comment = comments[i]

                content_between_comments = self.extract_until_comments(start_comment)
                second_half = self.extract_between_comments(start_comment, None)
                page_html_data = content_between_comments

        elif soup.find_all(re.compile("^hr")):
            page_break_tags = soup.find_all(re.compile("^hr"))
            # print("PAGE BREAK found", len(page_break_tags))
            if len(page_break_tags) > 1:
                for i in range(len(page_break_tags[:1])):  # all pages, not slicing
                    start_page_break = page_break_tags[i]

                    content_between_page_breaks = self.extract_until_comments(
                        start_page_break
                    )  # , end_page_break)
                    second_half = self.extract_between_comments(start_page_break, None)
                    page_html_data = content_between_page_breaks

        else:
            second_half = None
            page_html_data = None

        return page_html_data, second_half


class ProcessText:
    def __init__(self) -> None:
        pass

    def clean_text(self, text):
        """fucntion to clean text to remove
        special symbols and new line characters"""
        # text = text.encode('utf-8').decode('utf-8')
        text = text.replace("\n", " ")
        text = clean(text, lower=False)
        return text

    def add_commas(self, text):
        """function to add commas to values in Tables,
        as they were stripped in extraction.
        this is needed to map them back to html in
        serach and replace time."""
        value = text[::-1]
        new = []
        for i, char in enumerate(value, start=1):
            if i < len(value) and i % 3 == 0:
                new.append(char)
                new.append(",")
            else:
                new.append(char)

        new = "".join(new)
        new = new[::-1]
        return new

    def extract_number_from_text(self, text):
        return text.split()[-1]


def post_process(decoded_string, predictions):
    """function to decode tokenized text to readable text &
    map text to output labels in 2 different lists."""
    reconstructed_sentence = ""
    reconstructed_labels = []

    for subtoken, label in zip(decoded_string, predictions):
        # Skip special tokens and subtoken markers
        if subtoken != "<pad>":
            if subtoken in ["<s>", "</s>"] or subtoken.startswith("Ġ"):
                reconstructed_sentence = (
                    reconstructed_sentence + " " + subtoken.replace("Ġ", "")
                )
                reconstructed_labels.append(label)
            else:
                reconstructed_sentence += subtoken
    return reconstructed_sentence, reconstructed_labels


def process_table_results(table_names, columns, inputs, outputs):
    """Post processing for tabel results to particular pattern."""
    table_outputs = []
    for table_name, column, row, tag in zip(table_names, columns, inputs, outputs):
        row_line, output_tag = row.replace(table_name, "").replace(column, ""), tag
        numbers, output_tag = ProcessText().extract_number_from_text(row_line), tag
        numbers, output_tag = ProcessText().add_commas(str(numbers)), output_tag
        table_outputs.append({numbers: output_tag})

    return table_outputs


def process_notes_results(inputs, outputs):
    """Post processing for Notes results to particular pattern."""
    Notes_outputs = []
    for inp_row, oup_row in zip(inputs, outputs):
        for inp_word, oup_word in zip(inp_row.split(), oup_row):
            if oup_word != "O":
                Notes_outputs.append({inp_word: oup_word})
    return Notes_outputs
