"""
Title: 
    Table Processing utilities

Description:
    This file has the functionalities required for table processing.

Takeaways:
    - Table processing is a Complex case.
    - one should run these functionalities alone in jupyter notebook with debug statements for better understanding.
    - remove empty columns, cleaning existing columns, remvoing double span column names.
    - stripping "," from values to normalize them.

Author: purnasai@soulpage
Date: 10-10-2023
"""


import os
import re
import nltk
import shutil
import warnings

import pandas as pd
import numpy as np
from ast import literal_eval

from bs4 import BeautifulSoup, Comment
from nltk.tokenize import sent_tokenize
from IPython.display import display
from .utils import extract_content_between_comments

nltk.download("punkt")
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger(__name__)


def read_text_file(text_file_path):
    """reads text file & converts each row to list"""
    with open(text_file_path, "r") as fp:
        text = fp.readlines()
    text = [literal_eval(text_row) for text_row in text]
    return text


def clean_results(results):
    # this is to avoid keys with single val i.e
    #  {'0': 'us-gaap:CommonStocksIncludingAdditionalPaidInCapital'},
    # with search and replace, it replaces where ever it sees 0 in html string
    # which is not good. so cleaning them is better.
    results = [pair for pair in results for k, v in pair.items() if len(k) > 1]
    return results


def get_text_outside_table(content_between_comments):
    """Function to get all the text outside the table in a page"""
    soup1 = BeautifulSoup(content_between_comments, "lxml")
    # Remove all the tables from the HTML content
    for table in soup1.find_all("table"):
        table.extract()

    # Remove all the elements with table-related tags from the HTML content
    for tag in ["tbody", "thead", "tfoot", "tr", "th", "td"]:
        for element in soup1.find_all(tag):
            element.extract()

    # Get the text outside the tables and table-related elements
    text_outside_tables = soup1.get_text()
    return text_outside_tables


def parse_text(text_outside_tables):
    """this function used to assign a label using the text outsidet the table.
    if the text outside is similar to the wanted table headings, then this page
    is labeled/filtered as the table we want out of all statemnts"""
    # clean the text
    text_outside_tables = text_outside_tables.replace("\n", " ")
    text_outside_tables = text_outside_tables.replace("\t", " ")
    text_outside_tables = text_outside_tables.replace("  ", " ")
    text_outside_tables = text_outside_tables.upper()

    wanted_tables = {
        "BALANCE": "CONDENSED CONSOLIDATED BALANCE SHEETS",
        "OPERATIONS": "CONSOLIDATED STATEMENTS OF OPERATIONS AND COMPREHENSIVE LOSS",
        "SHAREHOLDERS’": "CONSOLIDATED STATEMENTS OF CHANGES IN SHAREHOLDERS’ EQUITY",
        "CASH FLOW": "CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS",
        #  "INCOME":"CONDENSED CONSOLIDATED STATEMENTS OF INCOME",
        "EQUITY": "CONSOLIDATED STATEMENTS OF EQUITY",
    }

    if any(x in text_outside_tables for x in wanted_tables.keys()):
        for table_name_key, table_name_value in wanted_tables.items():
            if table_name_key in text_outside_tables:
                logger.info(f"2.2. Table found......{table_name_value}")
                return table_name_value


def get_table(tables):
    """Function to take only first occurence of table,
    clean it, remove $)][] and empty spaces. drops
    empty columns and rows."""
    df = tables[0]
    df = (
        df.replace("[\$,)]", "", regex=True)
        .replace("[(]", "-", regex=True)
        .replace("", np.nan, regex=True)
    )
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    # this step is important
    mask = (df.iloc[1:, :].isna()).all(axis=0)
    financial_statements = df.drop(df.columns[mask], axis=1).fillna("")
    financial_statements.reset_index(drop=True, inplace=True)
    financial_statments = financial_statements.T.reset_index(drop=True).T
    return financial_statements


def get_column_df(financial_statements):
    """This is to find out the multi-row spread
    column/header using the first column space."""
    target_row = 0
    for col in financial_statements:
        for row_index, row in enumerate(financial_statements[col]):
            if row != "":
                target_row = row_index
                break
        break
    logger.info(f"Target Row:{target_row}")

    target_row = target_row + 1  # target_row if target_row > 0 else
    column_df = financial_statements.iloc[:target_row, :]

    def merge_column_df(column_df):
        ### merge multirows columns into a single row column.
        new_col_df = {}
        for col in column_df.columns:
            new_col_df[col] = " ".join(column_df[col].values)

        new_col_df = pd.DataFrame(new_col_df.items()).T
        # take only last row as dataframe
        new_col_df = new_col_df.iloc[-1:]
        return new_col_df

    column_df = merge_column_df(column_df)
    column_df.replace("", np.nan, regex=True, inplace=True)
    column_df.dropna(axis=0, how="all", inplace=True)
    return column_df, target_row


def merge_df(column_df, body_df):
    ### combine the dataframe once we get column & body df
    if column_df is not None:
        final_df = pd.concat([column_df, body_df], axis=0)
        final_df.reset_index(drop=True, inplace=True)
    else:
        final_df = body_df
    return final_df


def drop_empty_cols(final_df):
    """Drop empty columns &
    and columns with if they have only 1 or 2 values filled
    """
    drop_cols = []
    for col in final_df.columns:
        column_vals = [True for val in final_df[col].values if len(str(val)) > 1]
        if len(column_vals) == 1 or len(column_vals) == 2:
            drop_cols.append(col)

    final_df.drop(drop_cols, axis=1, inplace=True)
    final_df.reset_index(drop=True, inplace=True)
    return final_df


def create_header(final_df):
    """fucntion to create first row as header
    and replace -/— in the later rows."""
    final_df = final_df.reset_index(drop=True)
    final_df.columns = final_df.iloc[0]
    final_df = final_df[1:]
    final_df = final_df.apply(lambda x: x.str.replace("-", "").replace("—", ""))
    return final_df


def clean_duplicate_columns(final_df):
    """Remove duplicate columns, get occurence of each column
    and remove column if it occured more than once."""
    try:
        final_df = final_df.T.drop_duplicates().T
        max_values = 0
        # iterate through columns in final_df
        for col in final_df.columns:
            # get columns with same name
            cols = [c for c in final_df.columns if col in c]
            if len(cols) > 1:
                # get max value of columns
                max_values = final_df[cols].max(axis=1)
                # drop columns
                final_df.drop(cols, axis=1, inplace=True)
                # add max value column
                final_df[col] = max_values
    except:
        new_header = final_df.iloc[:2].agg(" ".join)
        final_df.columns = new_header
        final_df = final_df.iloc[2:].reset_index(drop=True)
    return final_df


def get_excel_statements_tables(html_table):
    """function that combines all above functionalities"""
    tables = pd.read_html(str(html_table))
    financial_statements = get_table(tables)
    column_df, target_row = get_column_df(financial_statements)

    # since some tables might have columns extended in body
    body_df = financial_statements.iloc[target_row:, :]
    body_df = body_df.T.reset_index(drop=True).T
    final_df = merge_df(column_df, body_df)
    final_df = drop_empty_cols(final_df)
    # display(final_df)
    try:
        final_df = create_header(final_df)
        final_df = final_df.rename(columns={final_df.columns[0]: "index"})
        final_df = clean_duplicate_columns(final_df)
    except:
        final_df = final_df
    return final_df


def get_rows_in_table(html_table):
    total_rows = []
    for table_row in html_table.find_all("tr"):
        # if row has any text
        if table_row.get_text().strip():
            tds = table_row.find_all("td")
            row_text = []
            for td in tds:
                # if it also has span tag
                if td.find("span"):
                    text = td.find("span").get_text()
                    row_text.append(text)
                else:
                    # td.find("span").get_text() missed $162 and $174
                    # Accounts receivable  net of allowances of $162 and $174
                    # so below worked for us
                    text = td.get_text()
                    row_text.append(text)

            row_text = [val.replace("$", "") for val in row_text if val != "$"]
            if row_text:
                total_rows.append(row_text)

    row_lengths = [len(row) for row in total_rows]
    # get mode of row lengths
    mode = max(set(row_lengths), key=row_lengths.count)
    complete_rows = [row for row in total_rows if len(row) == mode]
    return complete_rows


def clean_rows_and_tags(complete_rows):
    new_rows = []
    for row in complete_rows:
        row = [val for val in row if len(val) > 1]
        if len(row) > 1:
            context = row[0]
            tags = []
            for tag in row[1:]:
                try:
                    val, usgaap_tag = tag.split()
                except:
                    val = tag
                    usgaap_tag = "Others"
                val = re.sub("[^0-9]", "", val)

                tags.append({val: usgaap_tag})
                context += " " + val

            new_row = [context, tags]
            new_rows.append(new_row)
    return new_rows


def save_rows_and_tags(rows, file_path):
    with open(file_path, "w") as f:
        for row in rows:
            f.write("%s\n" % row)


def convert_float_to_int(value):
    """fuction that tries to convert every value
    in html table to float from string"""
    try:
        return int(float(value))
    except ValueError:
        return value


def get_cleaned_tags_data(text, dataframe, statement_name):
    statement_name = statement_name.lower()
    total_df_context_rows = []
    total_table_columns = []
    for text_row in text:
        tag_text = text_row[0]
        tag_text = " ".join([token.strip() for token in tag_text.split()])
        tag_text = re.sub(r"[,():\-]", "", tag_text)
        tags = text_row[1]
        tags = {val: tag for tag_item in tags for val, tag in tag_item.items()}

        for df_row in range(dataframe.shape[0]):
            df_row_text = " ".join(
                [
                    str(convert_float_to_int(val))
                    for val in dataframe.iloc[df_row].values
                ]
            )
            df_row_text = re.sub(r"[,():\-]", "", df_row_text)

            df_row_text = " ".join(df_row_text.split())
            tag_text = " ".join(tag_text.split())

            if tag_text == df_row_text:
                match_record = dataframe.iloc[df_row]

                columns = []
                contexts = []
                context = " "
                for column_name, value in match_record.items():
                    value = str(convert_float_to_int(value))

                    if tags.get(value) == None:
                        context = value
                    else:  # tags.get(value) != None:
                        us_gaap_tag = tags.get(value)
                        context1 = (
                            context
                            + " "
                            + column_name
                            + " "
                            + value
                            + "=="
                            + us_gaap_tag
                        )
                        if context1:
                            contexts.append(context1)
                            columns.append(column_name)
                if contexts and columns:
                    total_df_context_rows.extend(contexts)
                    total_table_columns.extend(columns)
                break
    total_df_context_rows = [
        " ".join([statement_name, row]) for row in total_df_context_rows
    ]
    return total_df_context_rows, total_table_columns


def save_html_statements_tables(html_data, save_path, type="10-Q"):
    """This function Detects Those statements table we want,
    and process them, saves them to the folder 'save_path'.
    """
    # create folder if not exists
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    os.makedirs(save_path)

    table_indx = 0
    soup = BeautifulSoup(html_data, "lxml")

    # Find all <!-- Field: Page; Sequence> tags
    comments = soup.find_all(
        string=lambda text: isinstance(text, Comment) and "Field: Page;" in text
    )

    # split html page by comments
    if comments:
        logger.info("2.1 Comment tags found...")
        if type == "10-K":
            logger.info("2.1.0. 10-K FILE, Using all Comments splits...")
            comments = comments[:]
        else:
            logger.info("2.1.1. 10-Q FILE, Using Only First 15 Comments splits...")
            comments = comments[:15]
        # Get the content between each pair of comments
        for i in range(
            len(comments) - 1
        ):  # looks only in the first 15 pages, since 10q
            start_comment = comments[i]
            end_comment = comments[i + 1]

            content_between_comments = extract_content_between_comments(
                start_comment, end_comment
            )
            html_tables = BeautifulSoup(content_between_comments, "lxml").find_all(
                "table"
            )

            if html_tables:
                text_outside_tables = get_text_outside_table(content_between_comments)
                table_name = parse_text(text_outside_tables)
                minimal_text = len(text_outside_tables)

                # should have atleast some text and less than 500 characters
                if minimal_text > 10 and minimal_text < 500:
                    logger.info(f"Found {table_name} ......")
                    if (
                        table_name
                        == "CONSOLIDATED STATEMENTS OF CHANGES IN SHAREHOLDERS’ EQUITY"
                    ):
                        for html_table in html_tables:
                            html_filename = (
                                f"page_comment_{table_name}_{table_indx}.html"
                            )
                            html_file_path = os.path.join(save_path, html_filename)

                            with open(html_file_path, "w", encoding="utf-8") as f:
                                f.write(str(html_table))

                                logger.info(
                                    f"page_comment_{table_name}_{table_indx}.html is saved"
                                )
                                table_indx += 1
                            try:
                                excel_file_path = html_file_path.replace(
                                    ".html", ".xlsx"
                                )
                                final_df = get_excel_statements_tables(html_table)
                                final_df.to_excel(excel_file_path, index=False)
                                logger.info(f"{excel_file_path} is saved")

                                rows_only = get_rows_in_table(html_table)
                                rows_and_tags = clean_rows_and_tags(rows_only)
                                save_rows_and_tags(
                                    rows_and_tags,
                                    html_file_path.replace(".html", ".txt"),
                                )
                                logger.info("txt file saved too....")
                            except:
                                logger.info("Couldn't Save TEXT and XLSX file... Please check")
                                pass

                    elif table_name:
                        html_filename = f"page_comment_{table_name}_{table_indx}.html"
                        html_file_path = os.path.join(save_path, html_filename)

                        with open(html_file_path, "w", encoding="utf-8") as f:
                            f.write(str(html_tables[0]))
                            logger.info(
                                f"page_comment_{table_name}_{table_indx}.html is saved"
                            )
                            table_indx += 1
                        try:
                            excel_file_path = html_file_path.replace(".html", ".xlsx")
                            final_df = get_excel_statements_tables(html_tables[0])
                            final_df.to_excel(excel_file_path, index=False)
                            logger.info(f"{excel_file_path} file saved")

                            rows_only = get_rows_in_table(html_tables[0])
                            rows_and_tags = clean_rows_and_tags(rows_only)
                            save_rows_and_tags(
                                rows_and_tags, html_file_path.replace(".html", ".txt")
                            )
                            logger.info("txt file saved too")
                        except:
                            logger.info("Couldn't Save TEXT and XLSX file... Please check")
                            pass

    else:
        # find all <hr style="page-break-after:always;"/>
        # page_breaks_divss = soup.find_all('div')
        page_break_tags = soup.find_all(re.compile("^hr"))
        if page_break_tags:
            logger.info("2.1 Header tags found...")
            # get the content between each paif of page_break_tags
            if type == "10-K":
                logger.info("2.1.2. 10-K FILE, Using all page breaks...")
                page_break_tags = page_break_tags[:]
            else:
                logger.info("2.1.3. 10-Q FILE, Using only First 15 Pages.....")
                page_break_tags = page_break_tags[:15]
            for i in range(len(page_break_tags) - 1):
                start_page_break = page_break_tags[i]
                end_page_break = page_break_tags[i + 1]

                content_between_page_breaks = extract_content_between_comments(
                    start_page_break, end_page_break
                )
                html_tables = BeautifulSoup(content_between_page_breaks).find_all(
                    "table"
                )

                if html_tables:
                    text_outside_tables = get_text_outside_table(
                        content_between_page_breaks
                    )
                    table_name = parse_text(text_outside_tables)

                    minimal_text = len(text_outside_tables)
                    if minimal_text < 500:
                        logger.info(f"Found {table_name} ......")
                        if (
                            table_name
                            == "CONSOLIDATED STATEMENTS OF CHANGES IN SHAREHOLDERS’ EQUITY"
                        ):
                            for html_table in html_tables:
                                html_filename = (
                                    f"page_headertag_{table_name}_{table_indx}.html"
                                )
                                html_file_path = os.path.join(save_path, html_filename)

                                with open(html_file_path, "w", encoding="utf-8") as f:
                                    f.write(str(html_table))
                                logger.info(f"{html_filename} is saved")
                                table_indx += 1
                                try:
                                    excel_file_path = html_file_path.replace(
                                        ".html", ".xlsx"
                                    )
                                    final_df = get_excel_statements_tables(html_table)
                                    final_df.to_excel(excel_file_path, index=False)
                                    logger.info(f"{excel_file_path} file saved")

                                    rows_only = get_rows_in_table(html_table)
                                    rows_and_tags = clean_rows_and_tags(rows_only)
                                    save_rows_and_tags(
                                        rows_and_tags,
                                        html_file_path.replace(".html", ".txt"),
                                    )
                                    logger.info("txt file saved too")
                                except:
                                    logger.info("Couldn't Save TEXT and XLSX file... Please check")
                                    pass

                        elif table_name:
                            html_filename = (
                                f"page_headertag_{table_name}_{table_indx}.html"
                            )
                            html_file_path = os.path.join(save_path, html_filename)

                            with open(html_file_path, "w", encoding="utf-8") as f:
                                f.write(str(html_tables[0]))
                            logger.info(f"{html_filename} is saved")
                            table_indx += 1
                            try:
                                excel_file_path = html_file_path.replace(
                                    ".html", ".xlsx"
                                )
                                final_df = get_excel_statements_tables(html_tables[0])
                                final_df.to_excel(excel_file_path, index=False)
                                logger.info(f"{excel_file_path} file saved")

                                rows_only = get_rows_in_table(html_tables[0])
                                rows_and_tags = clean_rows_and_tags(rows_only)
                                save_rows_and_tags(
                                    rows_and_tags,
                                    html_file_path.replace(".html", ".txt"),
                                )
                                logger.info("txt file saved too...")
                            except:
                                logger.info("Couldn't Save TEXT and XLSX file... Please check")
                                pass
                        print("\n")


## list of financial statements we are interested in
statement_names = [
    "BALANCE SHEET",
    "STATEMENTS OF CASH FLOW",
    "STATEMENTS OF EQUITY",
    "SHAREHOLDERS’ EQUITY",
    "OPERATIONS AND COMPREHENSIVE",
]


def arrange_rows_with_context(save_folder):
    """this function process table rows such it adds
    the context+columnname+value in to a single row
    """

    all_tables_data = []
    all_tables_columns = []
    table_names = []
    # iterate through all files
    for path, folders, files in os.walk(save_folder):
        if files:
            for file in files:
                if file.endswith(".html"):
                    html_file_path = os.path.join(path, file)
                    excel_file_path = html_file_path.replace(".html", ".xlsx")
                    text_file_path = html_file_path.replace(".html", ".txt")

                    if (
                        os.path.exists(excel_file_path)
                        and os.path.exists(html_file_path)
                        and os.path.exists(text_file_path)
                    ):
                        if (
                            os.path.getsize(excel_file_path)
                            and os.path.getsize(html_file_path)
                            and os.path.getsize(text_file_path)
                        ):
                            for statement_name in statement_names:
                                if statement_name in html_file_path:
                                    logger.info(
                                        f"2.3 Processing: {statement_name}, in {html_file_path}"
                                    )
                                    dataframe = pd.read_excel(excel_file_path)
                                    dataframe = dataframe.replace(
                                        np.nan, "", regex=True
                                    )

                                    text = read_text_file(text_file_path)
                                    logger.info(
                                        f"2.3.1 printing text, dataframe and table names {text},{dataframe},{statement_name}"
                                    )
                                    data, columns = get_cleaned_tags_data(
                                        text, dataframe, statement_name
                                    )

                                    all_tables_data.append(data)
                                    table_names.extend(
                                        [statement_name.lower()] * len(data)
                                    )
                                    all_tables_columns.extend(columns)
                        else:
                            logger.warning(
                                f"Empty files found, No content found for {statement_name} in {save_folder}"
                            )
                    else:
                        logger.warning(
                            f"All 3 Excel, html, text files not generate in {save_folder}"
                        )
        else:
            logger.warning(f"No File found in folder '{save_folder}'.")
    return all_tables_data, all_tables_columns, table_names