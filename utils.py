
import yaml




"""
    Utils file
    1 - load yaml config file
     
"""

def loadconfigYaml(path):
    with open(path, 'r') as stream:
        config_vars = yaml.safe_load(stream)
    
    return config_vars