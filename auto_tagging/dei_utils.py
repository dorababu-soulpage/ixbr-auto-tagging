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
import nltk
import warnings

from ast import literal_eval

from nltk import pos_tag
from nltk.tree import Tree
from nltk.chunk import conlltags2tree
from nltk.tokenize import sent_tokenize

from bs4 import BeautifulSoup, Comment

from .utils import clean_text
from .utils import extract_content_until_comment, read_html_file

nltk.download('punkt')
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger(__name__)


def collect_tokens(divs):
    """Pass a div tag, this function checks for the span
    table, tr tags and goes inside them, collects the text,
    splits, cleans them"""
    inputs = []

    for div in divs:
        if div.find("span") and not div.find("table"):
            logger.info("SPan tags found to collect text/tokens..")
            text = div.get_text()
                        
            # the below is creating \ax0 kind of symbols in text
            sentences = sent_tokenize(text)
            sentences = [clean_text(sentence) for sentence in sentences]

            ip_text = []
            for sentence in sentences:
                # so clean it
                sentence_tokens = sentence.split(" ")
                sentence_tokens = [clean_text(token) for token in sentence_tokens if len(clean_text(token)) >= 1]
                ip_text.append(sentence_tokens)

            inputs.extend(ip_text)
                    
        if not div.find("span"):
            if not div.find("tr"):
                logger.info("No span/table tags found, Extracting directly..")
                text = div.get_text()

                # this creates special symbols like xa08
                sentences = sent_tokenize(text)
                sentences = [clean_text(sentence) for sentence in sentences]

                ip_text = []
                for sentence in sentences:
                    # so clean it
                    sentence_tokens = sentence.split(" ")
                    sentence_tokens = [clean_text(token) for token in sentence_tokens if len(clean_text(token)) >= 1]
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

                        sentences = sent_tokenize(text)
                        sentences = [clean_text(sentence) for sentence in sentences]

                        ip_text = []
                        for sentence in sentences:
                            # so clean it
                            sentence_tokens = sentence.split(" ")
                            sentence_tokens = [clean_text(token) for token in sentence_tokens if len(clean_text(token)) >= 1]
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
                    sentences = sent_tokenize(text)
                    sentences = [clean_text(sentence) for sentence in sentences]

                    ip_text = []
                    for sentence in sentences:
                        # so clean it
                        sentence_tokens = sentence.split(" ")
                        sentence_tokens = [clean_text(token) for token in sentence_tokens if len(clean_text(token)) >= 1]
                        ip_text.append(sentence_tokens)
        
                    inputs.extend(ip_text)
        else:
            inputs = []
            logger.warning("No span/div/table tags found. Text collectiong Failed")
    
    return inputs


def split_page_and_extract_text(html_path):
    """Takes html file as input, read the data,
    checks for comment/header(hr) tag, splits
    the long html into parts. takes only first page
    & gets the text inside them."""
    html_data = read_html_file(html_path)
    soup = BeautifulSoup(html_data, 'lxml')

    # Find all <!-- Field: Page; Sequence> tags
    comments = soup.find_all(string=lambda text: isinstance(text, Comment) and 'Field: Page;' in text)

    # split html page by comments
    if comments:
        logger.info("Comments found as page break")
        for i in range(len(comments[:1])): # considering only cover page
            start_comment = comments[i]
            end_comment = comments[i + 1]
            content_between_comments = extract_content_until_comment(start_comment)
            page_html_data =  BeautifulSoup(content_between_comments, "lxml")

            divs = page_html_data.find_all("div")
            divs = divs[1:] # avoiding first div tag to avoid unncessary text: mmm-20230331.htm
            if divs and len(divs) > 10:
                logger.info("Only div tags found inside, collecting text...")
                inputs = collect_tokens(divs)
                total_rows = inputs

            elif divs and len(divs) < 10:
                logger.info("Only P tags found inside, collecting text...")
                ps = page_html_data.find_all("p")
                ps = ps[1:]
                inputs = collect_tokens(ps)

                tables = page_html_data.find_all("table")
                inputs1 = collect_tokens(tables)

                total_rows = inputs + inputs1

            else:
                logger.info("Only P tags found in Else block, collecting text...")
                ps = page_html_data.find_all("p")
                ps = ps[1:]

                inputs = collect_tokens(ps)

                tables = page_html_data.find_all("table")
                inputs1 = collect_tokens(tables)

                total_rows = inputs + inputs1

    elif soup.find_all(re.compile('^hr')):
        logger.info("Header tag found as page break")
        page_break_tags = soup.find_all(re.compile('^hr'))

        if len(page_break_tags)>1:
            for i in range(len(page_break_tags[:1])): # all pages, not slicing
                start_page_break = page_break_tags[i]
                end_page_break = page_break_tags[i + 1]

                content_between_page_breaks = extract_content_until_comment(start_page_break)
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


def remove_unpredicted_rows(total_reconstructed_sentence, total_reconstructed_labels):
    """Method to filter out the rows that were all tagged with "O" label"""
    new_words = []
    new_labels = []
    for in_row, out_row in zip(total_reconstructed_sentence, total_reconstructed_labels):
        words = in_row.strip().split(" ")[1:-1]
        labels = out_row[1:-1]
        if len(words) == len(labels):

            if list(set(labels))[0] != "O":
                new_words.extend(words)
                new_labels.extend(labels)

    return new_words, new_labels


def post_process_tags(tokens, tags):
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

    original_text
    return original_text

def format_processed_result(processed_result, total_given_input):
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

    return output_dict