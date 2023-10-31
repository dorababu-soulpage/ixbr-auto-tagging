import uuid
import string
import random

import logging
logger = logging.getLogger(__name__)


class OverwriteHtml():
    def __init__(self) -> None:
        pass
    
    def modify_coverpage(self, html_string, coverpage_output):
        """Function to use Coverpage/DEI results, search for the value in html
        and replace them with <font> tag"""
        logger.info("4.0. Overwriting HTML with COVERPAGE tags")

        ml_tags = list(coverpage_output.values())
        ml_tags = sum(ml_tags, [])
        unique_ml_tags = list({tup for tup in ml_tags})
        ml_tags = [[row[0], "dei:" + row[1]] for row in unique_ml_tags]
        uuid_result = uuid.uuid1()
        uuid_result = str(uuid_result).replace("-", "")
        ml_tags = [
            [
                row[0],
                row[1],
                f'<font data-autotag="true" id=xdx_90{random.choice(string.ascii_letters)}_e{row[1]}_{uuid_result}>{row[0]}</font>',
            ]
            for row in ml_tags
        ]

        # sample
        # result = html_string.replace("10-Q", '<font id="dei:DocumentType">10-Q</font>')
        html_string = html_string.replace("\n", " ")
        placeholders = {}

        for row in ml_tags:
            replaced_string = html_string.replace(">" + row[0] + "<", ">" + row[2] + "<")
            # Check if the string was modified by the first replacement
            if replaced_string != html_string:
                html_string = replaced_string
            else:
                # If the first replacement didn't work, apply the second replacement
                # Directly replacing is conflicting with the html format.
                # hence using place holder to replace all of them later.
                placeholder = f"__placeholder_{random.randint(1000, 9999)}__"
                html_string = html_string.replace(row[0], placeholder)
                placeholders[placeholder] = row[2]

        for placeholder, replacement in placeholders.items():
            html_string = html_string.replace(placeholder, replacement)

        return html_string


    def modify_statement_tabels(self, second_half, Table_output):
        """Function to use Table results, search for the value in html
        and replace them with <font> tag"""
        logger.info("4.1. Overwriting HTML with TABLE tags")

        Table_output1 = [list(list(dict_item.items())[0]) for dict_item in Table_output]
        # this only check for unique pairs. removes if both values in 2 pairs are same
        unique_lists = set(tuple(sublist) for sublist in Table_output1)
        unique_Table_output1 = [list(sublist) for sublist in unique_lists]

        unique_Table_output2 = []
        unique_vals = []
        for pair in unique_Table_output1:
            if pair[0] not in unique_vals:
                unique_vals.append(pair[0])
                unique_Table_output2.append(pair)

        uuid_result = uuid.uuid1()
        uuid_result = str(uuid_result).replace("-", "")
        Table_output1 = [
            [
                row[0],
                row[1],
                f'<font data-autotag="true" id=xdx_90{random.choice(string.ascii_letters)}_e{row[1]}_{uuid_result}>{row[0]}</font>',
            ]
            for row in unique_Table_output2
        ]

        second_half = second_half.replace("\n", " ")

        placeholders = {}
        # for row in Table_output1:
        #     second_half = second_half.replace(">"+row[0]+"<", ">"+row[2]+"<")
        for row in Table_output1:
            replaced_second_half = second_half.replace(
                ">" + row[0] + "<", ">" + row[2] + "<"
            )
            if replaced_second_half != second_half:
                second_half = replaced_second_half
            else:
                # replacing here itself might conflict with html like we had conflict in Coverpage
                # So better to use placeholders.
                placeholder = f'aa__placeholder_{str(random.randint(0, 9))+random.choice(string.ascii_letters)+str(random.randint(0, 9))+"-"+random.choice(string.ascii_letters)+"az"}__aa'
                second_half = second_half.replace(row[0], placeholder)
                placeholders[placeholder] = row[2]

        for placeholder, replacement in placeholders.items():
            second_half = second_half.replace(placeholder, replacement)

        return second_half


    def modify_notespages(self, second_half, Notes_output, table_output_values):
        """Function to use Notes results, search for the value in html
        and replace them with <font> tag"""
        logger.info("4.2. Overwriting HTML with NOTES tags")

        ml_tags1 = [tuple(t.items())[0] for t in Notes_output]
        ml_tags1 = [[row[0], "us-gaap:" + row[1]] for row in ml_tags1]
        uuid_result = uuid.uuid1()
        uuid_result = str(uuid_result).replace("-", "")
        ml_tags1 = [
            [
                row[0],
                row[1],
                f'<font data-autotag="true" id=xdx_90{random.choice(string.ascii_letters)}_e{row[1]}_{uuid_result}>{row[0]}</font>',
            ]
            for row in ml_tags1
            if row[0] not in table_output_values
        ]  # to avoid duplicating labels

        for row in ml_tags1:
            second_half = second_half.replace(">" + row[0] + "<", ">" + row[2] + "<")
        return second_half
