import json
import os
import sys
from src.preprocessing import generate_df_testdf
import src.baseline as baseline
sys.path.insert(0, 'src')
from src import eda_5types_dataset as eda
def main(targets):

    
    if 'test' in targets:
        test_crf = json.load(open('config/test.json'))
        
        BoG = baseline.build_model(test_crf['data_path'],test_crf['save_path'], model = 'BoG')
        Tfidf = baseline.build_model(test_crf['data_path'],test_crf['save_path'], model = 'Tfidf')
        
        print('==========================================')
        print('Finish test Target')
        print('==========================================')
        
        
    
    
    eda_config = json.load(open('config/eda.json'))
    if 'eda' in targets:
        eda.generate_stats('test', **eda_config)
        
        
if __name__ == '__main__':
    main(sys.argv[1:])
