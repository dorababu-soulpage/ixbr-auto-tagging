import boto3
import io, os, requests
from pathlib import Path
from decouple import config
from urllib.parse import urlsplit
from bs4 import BeautifulSoup
from utils import (
    get_db_record,
    update_db_record
)
from threading import Thread
from auto_tagging.tagging import auto_tagging
from flask import Flask, request, redirect, url_for


app = Flask(__name__)

storage_dir = "data"
base_dir = Path().absolute()
# create data directory if not exits
Path(f"{base_dir}/{storage_dir}").mkdir(parents=True, exist_ok=True)
storage_dir = Path(storage_dir).absolute()



def auto_tagging_thread(file_id:int):
    record = get_db_record(file_id = file_id)
    html = record.get("url", "")
    output_dir = f"{storage_dir}/html/{Path(html).stem}".replace("_","-")
    # create viewer folder
    Path(f"{output_dir}").mkdir(parents=True, exist_ok=True)
    filename = f"{output_dir}/{Path(html).stem}.html"
        # Send an HTTP GET request to the URL
    response = requests.get(html)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Get the content from the response
        html_content = response.text

        # Write the content to a local file
        with open(filename, "w") as file:
            file.write(html_content)
    
        output_html = "auto_tagging(filename)"
        try:
            with open(output_html, "rb") as file:
                body = io.BytesIO(file.read())
                # Parse the URL to extract the path
                parsed_url = urlsplit(output_html)
                # Get the filename from the path using pathlib
                path = Path(parsed_url.path)
                filename = path.name
                url = s3_uploader(name=filename, body=body)
                update_db_record(file_id, {"auto_tagging_url":url})
        except Exception as e:
            return {"error": "auto_tagging_html file is not generated"}, 400
    return {"error": "html file is not found"}, 400

@app.route("/api/auto-tagging", methods=["POST"])
def auto_tagging_view():
    
    file_id = request.json.get("file_id", None)
    # run process in background
    thread = Thread(target=auto_tagging_thread, args=(file_id,))
    thread.start()
    return {"message":"We will notify you once auto tagging is done."} , 200



if __name__ == "__main__":
    port = config("PORT")
    app.run(host="0.0.0.0", port=port, debug=True)