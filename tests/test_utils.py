import pytest
from auto_tagging.utils import FileManager




def test_load_yaml():
    model_config = FileManager().load_yaml("/home/ubuntu/auto-tagging/config.yaml")
    assert model_config["IXBRL_MODELS"]["Tokenizer"] == "soleimanian/financial-roberta-large-sentiment"
    assert model_config["IXBRL_MODELS"]["Dei_Model"] == "Models1/DEI_Model"
    assert model_config["IXBRL_MODELS"]["Notes_Model"] == "Models1/Notes_Model"

def test_read_html_file():
    with pytest.raises(FileNotFoundError):
        FileManager().read_html_file("dummy/path/to/file.html")
    



