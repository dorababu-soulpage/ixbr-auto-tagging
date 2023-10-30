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

nltk.download("punkt")

# utility imports
from .utils import (
    read_html_file,
    split_input_html,
    modify_coverpage,
    process_table_results,
    modify_statement_tabels,
    process_notes_results,
    modify_notespages,
)
from .dei_utils import (
    split_page_and_extract_text,
    remove_unpredicted_rows,
    post_process_tags,
    format_processed_result,
)
from .table_utils import (
    save_html_statements_tables,
    arrange_rows_with_context,
    clean_results,
)
from .notes_utils import get_NER_Data, clean_notes_outputs

# ml model imports
from .modelling import Xbrl_Tag
from .table_modelling import predict_table_tags

# get current date
current_date = datetime.date.today()
current_date = current_date.strftime("%d-%m-%Y")

# logging.basicConfig(filemode=)
logging.basicConfig(
    filename=os.path.join("logs", f"app_log_{current_date}.log"),
    filemode="w",
    format="%(asctime)s - %(levelname)s- %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)


def auto_tagging(html_file, html_type):
    xbrl_tag = Xbrl_Tag()
    html_path = html_file
    parent_dir = os.path.dirname(html_path)
    dest_path = os.path.join(parent_dir, "copied_html.html")
    shutil.copy(html_path, dest_path)
    logging.info(f"0. FIle type received is {html_type}")

    ### 1.COVERPAGE
    logging.info("1.Processing Cover page...............")
    total_rows = split_page_and_extract_text(html_path)

    logging.info("1.1 Started predicting DEI tags.....")
    original_inputs, inputs, outputs = xbrl_tag.predict_dei_tags(total_rows)
    inputs, outputs = remove_unpredicted_rows(inputs, outputs)
    logging.info("1.2. Started Post processing DEI Tags.....")
    processed_result = post_process_tags(inputs, outputs)
    coverapge_results = format_processed_result(processed_result, original_inputs)
    print("Coverpage results lenght:", len(coverapge_results))
    logging.info("1.3. Completed DEI tags sucessfully")
    logging.info(f"{coverapge_results}")

    ### 2.TABLE
    logging.info("2.Processing Statement tables...............")
    normalized_path = os.path.normpath(html_path)
    file_name = os.path.basename(normalized_path)
    folder, _ = file_name.split(".")

    # folder to save
    save_folder = "Table_raw_results"
    html_data = read_html_file(html_path)
    save_path = os.path.join(save_folder, folder)

    save_html_statements_tables(html_data, save_path, html_type)
    data, columns, table_names = arrange_rows_with_context(save_path)
    logging.info(f"{data, columns, table_names}")

    inputs, outputs = predict_table_tags(data)
    for i,j in zip(inputs, outputs):
        logging.info(f"{i},{j}")

    table_outputs = process_table_results(table_names, columns, inputs, outputs)
    table_outputs = clean_results(table_outputs)
    print("length of table results:", len(table_outputs))
    logging.info(f"{table_outputs}")

    ### 3.Notes
    logging.info("3. Processing Notes in Filings.......")
    logging.info("3.0.Processing Entire HTML instead of NOTES Sections.")
    # TODO: Should only run Notes section instead of Entire HTML.

    html_data = read_html_file(html_path)
    input_data = get_NER_Data(html_data)

    logging.info("3.2. starting predicting Notes tags....")
    inputs, outputs = xbrl_tag.predict_notes_tags(input_data)

    logging.info("3.3. Removes predicted sentences with 'O' tag entirely")
    inputs, outputs = clean_notes_outputs(inputs, outputs)
    table_output_values = [key for row in table_outputs for key, val in row.items()]
    Notes_outputs = process_notes_results(inputs, outputs)
    Notes_outputs = clean_results(Notes_outputs)
    print("length of notes results:", len(Notes_outputs))
    logging.info(f"{Notes_outputs}")

    # #######################################################
    # #########Overwrite HTML file###########################
    # #######################################################
    logging.info("4. Overwriting HTML File with ML Model Results..")
    coverapge, other_pages = split_input_html(dest_path)
    html_string, other_pages1 = copy.deepcopy(coverapge), copy.deepcopy(other_pages)

    html_string = modify_coverpage(html_string, coverapge_results)
    other_pages2 = modify_statement_tabels(other_pages1, table_outputs)
    other_pages3 = modify_notespages(other_pages2, Notes_outputs, table_output_values)

    print(len(html_string), len(other_pages3))
    final_result = html_string + other_pages3
    print(len(final_result))
    final_result = BeautifulSoup(final_result)
    
    logging.info("4.3 Printing predicted Tags summary before SAVING HTML...")
    logging.info("TOTAL TAGS:\nCoverpage results length: {}\nTable results length: {}\nNotes results length: {}".format(
                                                                        len(coverapge_results),
                                                                        len(table_outputs),
                                                                        len(Notes_outputs),
                                                                        ))

    dest_path = os.path.join(parent_dir, f"auto_tagging_{os.path.basename(html_file)}")
    with open(dest_path, "wb") as file:
        file.write(final_result.encode("utf-8"))
    logging.info("5. Finally FILE Saved")
    logging.shutdown()
    return dest_path
