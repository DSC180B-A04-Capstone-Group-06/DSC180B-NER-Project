import json
import os
import sys
from src.preprocessing import generate_df_testdf

def main(targets):

    preprocess_crf = json.load(open('config/preprocess_cfg.json'))
    eda_crf = json.load(open('config/eda.json'))

if __name__ == '__main__':
    main(sys.argv[1:])
