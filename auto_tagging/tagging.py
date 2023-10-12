import os
import copy
import nltk
import shutil
import logging
import datetime

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup, Comment
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

# utility imports
from .utils import read_html_file, split_input_html, modify_coverpage, add_commas, extract_number_from_text, process_table_results,modify_statement_tabels,\
            process_notes_results, modify_notespages
from .dei_utils import split_page_and_extract_text, remove_unpredicted_rows, post_process_tags, format_processed_result
from .table_utils import save_html_statements_tables, arrange_rows_with_context
from .notes_utils import get_NER_Data, clean_notes_outputs


# ml model imports
from .modelling import Xbrl_Tag
from .table_modelling import predict_table_tags


# get current date
current_date = datetime.date.today()
current_date = current_date.strftime("%d-%m-%Y")

# logging.basicConfig(filemode=)
logging.basicConfig(filename= f"app_log_{current_date}.log", filemode="w", 
                    format='%(asctime)s - %(levelname)s- %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)

def auto_tagging(html_file):
    # instantiating ML Model Main class
    xbrl_tag = Xbrl_Tag()
    # html_path = r"D:\Purna_Office\Soulpage_New\Apex_ixbrl\Final_App\Southern First Bancshares, Inc\Raw Input\sfst4220081-10q.htm" #sfst4220081-10q.htm
    html_path = html_file
    parent_dir = os.path.dirname(html_path)
    dest_path = os.path.join(parent_dir, "copied_html.html")
    shutil.copy(html_path, dest_path)

    ### 1.COVERPAGE
    logging.info('1.Processing Cover page...............')
    total_rows = split_page_and_extract_text(html_path)

    logging.info("1.1 Started predicting DEI tags.....")
    original_inputs, inputs, outputs = xbrl_tag.predict_dei_tags(total_rows)
    inputs, outputs = remove_unpredicted_rows(inputs, outputs)

    logging.info("1.2. Started Post processing DEI Tags.....")
    processed_result = post_process_tags(inputs, outputs)
    coverapge_results = format_processed_result(processed_result, original_inputs)
    logging.info("1.3. Completed DEI tags sucessfully")

    ### 2.TABLE
    logging.info('2.Processing Statement tables...............')
    normalized_path = os.path.normpath(html_path)
    file_name = os.path.basename(normalized_path)
    folder, _ = file_name.split(".")

    # folder to save
    save_folder = "Table_raw_results"
    html_data = read_html_file(html_path)
    save_path = os.path.join(save_folder, folder)

    save_html_statements_tables(html_data, save_path)
    data, columns, table_names = arrange_rows_with_context(save_path)
    inputs, outputs = predict_table_tags(data)
    table_outputs = process_table_results(table_names, columns, inputs, outputs)

    ### 3.Notes
    logging.info("3. Processing Notes in Filings.......")
    html_data = read_html_file(html_path)
    input_data = get_NER_Data(html_data)

    logging.info("3.2. starting predicting Notes tags....")
    inputs, outputs = xbrl_tag.predict_notes_tags(input_data)

    logging.info("3.3. Removes predicted sentences with 'O' tag entirely")
    inputs, outputs = clean_notes_outputs(inputs, outputs)
    table_output_values = [key for row in table_outputs for key,val in row.items()]
    Notes_outputs = process_notes_results(inputs, outputs)
    logging.shutdown()

    # #######################################################
    # #########Overwrite HTML file###########################
    # #######################################################
    coverapge, other_pages = split_input_html(dest_path)
    html_string, other_pages1 = copy.deepcopy(coverapge), copy.deepcopy(other_pages)

    html_string = modify_coverpage(html_string, coverapge_results)
    other_pages2 = modify_statement_tabels(other_pages1, table_outputs)
    other_pages3 = modify_notespages(other_pages2, Notes_outputs, table_output_values)

    print(len(html_string), len(other_pages3))
    final_result = html_string+other_pages3
    print(len(final_result))
    final_result = BeautifulSoup(final_result)

    dest_path = os.path.join(parent_dir, os.path.basename(html_file))
    with open(dest_path, "wb") as file:
        file.write(final_result.encode("utf-8"))

    return dest_path