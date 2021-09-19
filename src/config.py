"""
__author__: Le Trung Phong
__email__ = letrungphong95@gmail.com
"""
from typing import Dict
import yaml 

class Config(Dict):
    """This class represents the configuration for our project 
    """
    def __init__(self, file_path):
        "Initialize parameters"
        self.set_parameters(file_path)

    def set_parameters(self, file_path):
        "Load the yaml file and set attribute"
        with open(file_path, 'r') as f: 
            cfg = yaml.full_load(f)
        for k, v in cfg.items():
            setattr(self, k, v)
