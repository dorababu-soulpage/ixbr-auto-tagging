import pytest
import warnings
warnings.filterwarnings("ignore")

from auto_tagging.utils import FileManager, ProcessText

def test_load_yaml():
    model_config = FileManager().load_yaml("/home/ubuntu/auto-tagging/config.yaml")
    assert model_config["IXBRL_MODELS"]["Tokenizer"] == "soleimanian/financial-roberta-large-sentiment"
    assert model_config["IXBRL_MODELS"]["Dei_Model"] == "Models1/DEI_Model"
    assert model_config["IXBRL_MODELS"]["Notes_Model"] == "Models1/Notes_Model"

def test_read_html_file():
    with pytest.raises(FileNotFoundError):
        FileManager().read_html_file("dummy/path/to/file.html")
    


class TestProcessText:
    """class name should have "Test" starting for
    pytest to identify the class.
    
    __init__ dunders wont work here"""
    
    def setup_method(self, method):
        print(f"Setting up {method}")
        self.text_process = ProcessText()
        self.sample_text = "Soulpage\x92s is \n an AI\xa0 Organization"
        self.cleaned_sample_text = "Soulpage's is an AI Organization"

        self.random_no = "900003421"
        self.random_no_readable= "900,003,421"

    def teardown_method(self, method):
        print(f"Tearing down {method}")
        print(self.text_process.clean_text(self.sample_text))
        print(self.text_process.add_commas(self.random_no))

    def test_clean_test(self):
        assert self.text_process.clean_text(self.sample_text) == self.cleaned_sample_text
        assert self.text_process.add_commas(self.random_no) == self.random_no_readable



