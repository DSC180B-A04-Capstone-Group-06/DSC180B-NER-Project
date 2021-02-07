import pandas as pd
import os
import numpy as np

def generate_df_testdf(path, save_p):
    
    data_folder = path + 'News Articles/'
    summary_folder = path + 'Summaries/'
    entries = os.listdir(data_folder)


    all_data = {}
    for i in entries:
        temp = []
        folder_path = data_folder + i +'/'
        file_lst = os.listdir(folder_path)
        for j in file_lst:
            if j != '.ipynb_checkpoints':
                with open(folder_path +j, 'r', errors='ignore') as file:
                    temp.append(file.read().replace('\n', ''))
        all_data[i] = temp

    all_sum = {}
    for i in entries:
        temp = []
        folder_path = summary_folder + i +'/'
        file_lst = os.listdir(folder_path)
        for j in file_lst:
            if j != '.ipynb_checkpoints':
                with open(folder_path +j, 'r', errors='ignore') as file:
                    temp.append(file.read().replace('\n', ''))
        all_sum[i] = temp



    total_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for i in np.arange(len(entries)):
        if i == 0:
            total_df = pd.DataFrame.from_dict(all_data['business'])
            total_df['type'] = 'business'
            total_df['summary'] = all_sum['business']
            total_df['type_code'] = i+1
            
            
            index = np.random.choice(total_df.shape[0],10)
            test_df = total_df.loc[index]
        else:
            temp_df = pd.DataFrame.from_dict(all_data[entries[i]])
            temp_df['type'] = entries[i]
            temp_df['summary'] = all_sum[entries[i]]
            temp_df['type_code'] = i+1
            total_df =pd.concat([total_df, temp_df], axis=0)
            
            index = np.random.choice(temp_df.shape[0], 10)
            temp_test_df = temp_df.loc[index]
            test_df =pd.concat([test_df,temp_test_df] , axis=0) 
            
    total_df.columns = ['text','type','summary','type_code']
    total_df = total_df.reset_index(drop = True)
    total_df.to_csv(os.path.join(save_p+'all_data.csv'))

    test_df.columns = ['text','type','summary','type_code']
    test_df = test_df.reset_index(drop = True)
    test_df.head()
    test_df.to_csv(os.path.join(save_p +'test.csv'))

    return 'Done'