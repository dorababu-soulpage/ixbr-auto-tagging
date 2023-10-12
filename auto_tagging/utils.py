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
import random
import string
import uuid 

from bs4 import BeautifulSoup, Comment


import warnings
warnings.filterwarnings("ignore")

def read_html_file(html_path):
    """fucntion to read html file"""
    with open(html_path, 'r', encoding="unicode-escape") as file:
        html_data = file.read()
    return html_data

def extract_content_until_comment(start_comment):
    """function to get html code until the page break"""
    content = ""
    curr = start_comment.find_previous_sibling()
    while curr:
        content = str(curr) + content
        if curr.name == 'comment':
            break
        curr = curr.find_previous_sibling()
    return content


def extract_content_between_comments(start_comment, end_comment):
    """fucntion to get html code from one page start to same page end"""
    content = ""
    curr = start_comment.next_sibling
    while curr and curr != end_comment:
        content += str(curr)
        curr = curr.next_sibling
    return content


def clean_text(text):
    """fucntion to clean text to remove 
    special symbols and new line characters"""
    text = text.replace("\x92", "'")
    text = text.replace("\xa0", " ")
    text = text.replace("\n"," ")
    return text


def post_process(decoded_string, predictions):
    """function to decode tokenized text to readable text &
    map text to output labels in 2 different lists."""
    reconstructed_sentence = ""
    reconstructed_labels = []

    for subtoken,label in zip(decoded_string, predictions):
        # Skip special tokens and subtoken markers
        if subtoken != "<pad>":
            if subtoken in ["<s>", "</s>"]  or subtoken.startswith("Ġ"):
                reconstructed_sentence =  reconstructed_sentence+" "+subtoken.replace("Ġ", "")
                reconstructed_labels.append(label)
            else:
                reconstructed_sentence += subtoken
    return reconstructed_sentence, reconstructed_labels


def split_input_html(html_path):
    """this is to split the input html that user uploads in to platform.
    the same html is overwrittern with output tags. Simple split is used, 
    rather than the complex split we did in fucntions below
    - split_page_and_extract_text
    - collect_tokens
    in dei_utils.py file"""
    with open(html_path, 'r', encoding="unicode-escape") as file:
        html_data = file.read()
    
    soup = BeautifulSoup(html_data, 'html.parser', ) #preserve_whitespace=True
    # Find all <!-- Field: Page; Sequence> tags
    comments = soup.find_all(string=lambda text: isinstance(text, Comment) and 'Field: Page;' in text)

    # split html page by comments
    if comments:
        print("Comments found")
        for i in range(len(comments[:1])): # all pages, not slicing
            start_comment = comments[i]

            content_between_comments = extract_content_until_comment(start_comment)
            second_half = extract_content_between_comments(start_comment, None)
            page_html_data =  content_between_comments

    elif soup.find_all(re.compile('^hr')):
        page_break_tags = soup.find_all(re.compile('^hr'))
        print("PAGE BREAK found", len(page_break_tags))
        if len(page_break_tags)>1:
            for i in range(len(page_break_tags[:1])): # all pages, not slicing
                start_page_break = page_break_tags[i]

                content_between_page_breaks = extract_content_until_comment(start_page_break)#, end_page_break)
                second_half = extract_content_between_comments(start_page_break, None)
                page_html_data = content_between_page_breaks

    else:
        second_half = None
        page_html_data = None
    
    return page_html_data, second_half

def add_commas(sttt):
    """function to add commas to values in Tables,
    as they were stripped in extraction. 
    this is needed to map them back to html in 
    serach and replace time."""
    sttt = sttt[::-1]
    new = []
    for i,char in enumerate(sttt, start=1):
        if i< len(sttt) and i%3==0:
            new.append(char)
            new.append(",")
        else:
            new.append(char)

    new = "".join(new)
    new = new[::-1]
    return new


def extract_number_from_text(text):
    # Use a regular expression to find an integer that is part of a word or phrase
    match = re.search(r'\b\d+\b', text)

    if match:
        # Extract the matched integer
        integer_value = int(match.group())
        return integer_value
    else:    
        return text

def process_table_results(table_names, columns, inputs, outputs):
    """Post processing for tabel results to particular pattern."""
    table_outputs = []
    for table_name,column,row,tag in zip(table_names, columns, inputs, outputs):
        # print(table_name, column, row, tag)
        # print(table_name, column, row,  row.replace(table_name,"").replace(column,""), tag)
        row_line,output_tag = row.replace(table_name,"").replace(column,""), tag
        numbers,output_tag  = extract_number_from_text(row_line), tag
        numbers,output_tag  = add_commas(str(numbers)), output_tag
        table_outputs.append({numbers:output_tag})
    return table_outputs

def process_notes_results(inputs, outputs):
    """Post processing for Notes results to particular pattern."""
    Notes_outputs = []
    for inp_row,oup_row in zip(inputs, outputs):
        # print(inp_row, oup_row)
        for inp_word, oup_word in zip(inp_row.split(), oup_row):
            if oup_word != "O":
                Notes_outputs.append({inp_word: oup_word})
    return Notes_outputs

def modify_coverpage(html_string, coverpage_output):
    """Function to use Coverpage/DEI results, search for the value in html
    and replace them with <font> tag"""
    # cover page tags
    # place_holder  = '<font id="dei:DocumentType">10-Q</font>'
    ml_tags = list(coverpage_output.values())
    ml_tags = sum(ml_tags,[])
    ml_tags = [[row[0], "dei:"+row[1]] for row in ml_tags]
    # ml_tags = [[row[0], row[1], f'<font data-autotag="true" id=xdx_90{random.choice(string.ascii_letters)}_e{row[1]}_{uuid.uuid1()}>{row[0]}</font>'] for row in ml_tags]
    ml_tags = [[row[0], row[1], f'<font data-autotag="true" id=xdx_90{random.choice(string.ascii_letters)}_e{row[1]}_{uuid.uuid1()}>{row[0]}</font>'] for row in ml_tags]
    
    # sample
    # result = html_string.replace("10-Q", '<font id="dei:DocumentType">10-Q</font>')

    for row in ml_tags:
        # print(row[0], row[2])
        html_string = html_string.replace(">"+row[0]+"<", ">"+row[2]+"<")
    return html_string


def modify_statement_tabels( second_half, Table_output):
    """Function to use Table results, search for the value in html
    and replace them with <font> tag"""
    # table 
    Table_output1 = [list(list(dict_item.items())[0]) for dict_item in Table_output]
    Table_output1 = [[row[0], row[1], f'<font data-autotag="true" id=xdx_90{random.choice(string.ascii_letters)}_e{row[1]}_{uuid.uuid1()}>{row[0]}</font>'] for row in Table_output1]
    
    for row in Table_output1:
        second_half = second_half.replace(">"+row[0]+"<", ">"+row[2]+"<")

    return second_half

def modify_notespages(second_half, Notes_output, table_output_values):
    """Function to use Notes results, search for the value in html
    and replace them with <font> tag"""
    # Notes
    ml_tags1 = [tuple(t.items())[0] for t in Notes_output]
    ml_tags1 = [[row[0], "us-gaap:"+row[1]] for row in ml_tags1]
    ml_tags1 = [[row[0], row[1], f'<font data-autotag="true" id=xdx_90{random.choice(string.ascii_letters)}_e{row[1]}_{uuid.uuid1()}>{row[0]}</font>'] for row in ml_tags1 if row[0] not in table_output_values] # to avoid duplicating labels

    for row in ml_tags1:
        second_half = second_half.replace(">"+row[0]+"<", ">"+row[2]+"<")
    return second_half