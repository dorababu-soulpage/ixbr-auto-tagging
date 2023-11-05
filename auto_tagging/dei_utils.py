"""
Title: 
    DEI Utilities

Description:
    This file has all the functionalities/utilities required for Coverpage processing.

Takeaways:
    - It even processes the Tables in the Coverpage.
    - Check boxes are not properly parserd. to future scope.

Author: purnasai@soulpage
Date: 10-10-2023
"""

import re
import bs4
import nltk
import warnings
import logging

from nltk import pos_tag
from nltk.tree import Tree
from nltk.chunk import conlltags2tree
from nltk.tokenize import sent_tokenize

from typing import List, Dict
from bs4 import BeautifulSoup, Comment
from .utils import FileManager, HtmlContent, ProcessText

nltk.download('punkt')
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

processtext = ProcessText()

def collect_tokens(divs:list) -> list: 
    """Pass a div tag, this function checks for the span
    table, tr tags and goes inside them, collects the text,
    splits, cleans them"""
    inputs = []

    logging.info('1.0. collecting tokens...............')
    for div in divs:
        if div.find("span") and not div.find("table"):
            logger.info("SPan tags found to collect text/tokens..")
            text: str = div.get_text()
                        
            # the below is creating \ax0 kind of symbols in text
            sentences: List[str] = sent_tokenize(text)
            sentences: List[str] = [processtext.clean_text(sentence) for sentence in sentences]

            ip_text = []
            for sentence in sentences:
                # so clean it
                sentence_tokens = sentence.split(" ")
                sentence_tokens = [processtext.clean_text(token) for token in sentence_tokens if len(processtext.clean_text(token)) >= 1]
                ip_text.append(sentence_tokens)

            inputs.extend(ip_text)
                    
        if div.find("span") and div.find("table"):
            # if span and table tag found then extract table information
            table = div.find("table")

            for table_row in table.find_all("tr"):
                row = []
                for td in table_row.find_all("td"):
                    if td.get_text():
                        row.append(td.get_text())
                if row:
                    text = " ".join(row)

                    # the below is creating \ax0 kind of symbols in text
                    sentences: List[str] = sent_tokenize(text)
                    sentences: List[str] = [processtext.clean_text(sentence) for sentence in sentences]

                    ip_text = []
                    for sentence in sentences:
                        # so clean it
                        sentence_tokens = sentence.split(" ")
                        sentence_tokens = [processtext.clean_text(token) for token in sentence_tokens if len(processtext.clean_text(token)) >= 1]
                        ip_text.append(sentence_tokens)
        
                    inputs.extend(ip_text)
        
        if div.find_all("p"):
            # if <P> tag inside div tag
            for p in div.find_all("p"):
                text = p.get_text() 

                # this creates special symbols like xa08
                sentences: List[str] = sent_tokenize(text)
                sentences: List[str] = [processtext.clean_text(sentence) for sentence in sentences]

                ip_text = []
                for sentence in sentences:
                    # so clean it
                    sentence_tokens = sentence.split(" ")
                    sentence_tokens = [processtext.clean_text(token) for token in sentence_tokens if len(processtext.clean_text(token)) >= 1]
                    ip_text.append(sentence_tokens)
                inputs.extend(ip_text)
        
        if not div.find("span"):
            if not div.find("tr"):
                logger.info("No span/table tags found, Extracting directly..")
                text = div.get_text() 

                # this creates special symbols like xa08
                sentences: List[str] = sent_tokenize(text)
                sentences: List[str] = [processtext.clean_text(sentence) for sentence in sentences]

                ip_text = []
                for sentence in sentences:
                    # so clean it
                    sentence_tokens = sentence.split(" ")
                    sentence_tokens = [processtext.clean_text(token) for token in sentence_tokens if len(processtext.clean_text(token)) >= 1]
                    ip_text.append(sentence_tokens)
                inputs.extend(ip_text)
            
            elif div.find_all("tr"):
                # if table tag found
                for table_row in div.find_all("tr"):
                    row = []
                    for td in table_row.find_all("td"):
                        if td.get_text():
                            row.append(td.get_text())
                    if row:
                        text = " ".join(row)

                        sentences: List[str] = sent_tokenize(text)
                        sentences: List[str] = [processtext.clean_text(sentence) for sentence in sentences]

                        ip_text = []
                        for sentence in sentences:
                            # so clean it
                            sentence_tokens = sentence.split(" ")
                            sentence_tokens = [processtext.clean_text(token) for token in sentence_tokens if len(processtext.clean_text(token)) >= 1]
                            ip_text.append(sentence_tokens)
                        inputs.extend(ip_text)


        else:
            text = div.get_text()
            sentences: List[str] = sent_tokenize(text)
            sentences: List[str] = [processtext.clean_text(sentence) for sentence in sentences]

            ip_text = []
            for sentence in sentences:
                # so clean it
                sentence_tokens = sentence.split(" ")
                sentence_tokens = [processtext.clean_text(token) for token in sentence_tokens if len(processtext.clean_text(token)) >= 1]
                ip_text.append(sentence_tokens)
            inputs.extend(ip_text)
            # logger.warning("No span/div/table tags found. Text collectiong Failed")
    return inputs

def process_p_tags(page_html_data:bs4.BeautifulSoup) -> list:
    ps = page_html_data.find_all("p")
    ps = ps[1:]
    inputs = collect_tokens(ps)

    tables = page_html_data.find_all("table")
    inputs1 = collect_tokens(tables)

    total_rows = inputs + inputs1
    return total_rows

def split_page_and_extract_text(html_path) -> list:
    """Takes html file as input, read the data,
    checks for comment/header(hr) tag, splits
    the long html into parts. takes only first page
    & gets the text inside them."""
    html_data: str = FileManager().read_html_file(html_path)
    soup = BeautifulSoup(html_data, 'lxml')

    # Find all <!-- Field: Page; Sequence> tags
    comments: list = soup.find_all(string=lambda text: isinstance(text, Comment) and 'Field: Page;' in text)

    # split html page by comments
    if comments:
        logger.info("Comments found as page break")
        for i in range(len(comments[:1])): # considering only cover page
            start_comment = comments[i]
            content_between_comments = HtmlContent().extract_until_comments(start_comment)
            content_between_comments = content_between_comments.encode("utf-8")
            page_html_data =  BeautifulSoup(content_between_comments, "lxml", from_encoding="utf-8")

            divs = page_html_data.find_all("div")
            divs = divs[1:] # avoiding first div tag to avoid unncessary text: mmm-20230331.htm
            
            if divs and len(divs) > 10:
                logger.info("Only div tags found inside, collecting text...")
                inputs = collect_tokens(divs)
                total_rows = inputs
                
                if total_rows == []:
                    logger.info("No, all divs empty. found P tags inside, collecting text...")
                    total_rows = process_p_tags(page_html_data)

            elif divs and len(divs) < 10:
                    logger.info("less divs found. So finding P tags, collecting text...")
                    total_rows = process_p_tags(page_html_data)

            else:
                logger.info("Only P tags found in Else block, collecting text...")
                total_rows = process_p_tags(page_html_data)

    # split html by header tags.
    elif soup.find_all(re.compile('^hr')):
        logger.info("Header tag found as page break")
        page_break_tags = soup.find_all(re.compile('^hr'))

        if len(page_break_tags)>1:
            for i in range(len(page_break_tags[:1])): # Considering only coverpage
                start_page_break = page_break_tags[i]
                content_between_page_breaks = HtmlContent().extract_until_comments(start_page_break)
                page_html_data =  BeautifulSoup(content_between_page_breaks)

                divs = page_html_data.find_all("div")
                divs = divs[1:] # avoiding first div tag to avoid unncessary text: mmm-20230331.htm
                if divs:
                    inputs = collect_tokens(divs)
                    total_rows = inputs

                else:
                    ps = page_html_data.find_all("p")
                    ps = ps[1:]

                    inputs = collect_tokens(ps)
                    total_rows = inputs

    else:
        total_rows = []
        logger.warning("No Page break/comment found.")
    return total_rows


def remove_unpredicted_rows(total_reconstructed_sentence: list, total_reconstructed_labels: list):
    """Method to filter out the rows that were all tagged with "O" label"""
    new_words = []
    new_labels = []
    for in_row, out_row in zip(total_reconstructed_sentence, total_reconstructed_labels):
        words = in_row.strip().split(" ")[1:-1]
        labels = out_row[1:-1]
        
        if len(words) == len(labels):
            if len(words)>1 and len(labels)>1:
                # if list(set(labels))[0] != "O":
                    new_words.extend(words)
                    new_labels.extend(labels)

    return new_words, new_labels


def post_process_tags(tokens, tags) -> List[tuple]:
    """Method i.e final step in postprocessing in DEI tags"""
    # tag each token with pos
    pos_tags = [pos for token, pos in pos_tag(tokens)]

    # convert the BIO / IOB tags to tree
    conlltags = [(token, pos, tg) for token, pos, tg in zip(tokens, pos_tags, tags)]
    ne_tree = conlltags2tree(conlltags)

    # parse the tree to get our original text
    original_text = []
    for subtree in ne_tree:
        # skipping 'O' tags
        if type(subtree) == Tree:
            original_label = subtree.label()
            original_string = " ".join([token for token, pos in subtree.leaves()])
            original_text.append((original_string, original_label))

    return original_text

def format_processed_result(processed_result: List[tuple], total_given_input: list) -> Dict:
    """ This maps/places all values&their tags to the original row,
    i.e like below
    {'1025 Connecticut Avenue NW Suite 1000': [('1025 Connecticut Avenue',
    'EntityAddressAddressLine1'),
    ('NW', 'EntityAddressAddressLine2'),
    ('Suite 1000', 'EntityAddressAddressLine2')]}"""
    
    output_dict = {}
    for orig_row in total_given_input:
        for item in processed_result:
            if item[0] in orig_row and len(item[0])>1:
                
                if orig_row not in output_dict:
                    output_dict[orig_row] = [item]
                else:
                    output_dict[orig_row].append(item)

    for row in list(output_dict):
        if "filer" in row.lower():
            del output_dict[row]
    
    return output_dict