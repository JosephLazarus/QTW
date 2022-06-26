#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Description: Clustering with h2o.ai
# Author: Blanchard
# Created on: 2021-09-21
"""

############################################
###### Import required Python packages #####
############################################

import os
import pandas as pd
import datetime as dt
import re
import numpy as np
import logging
import logging.config
import copy
import math
import h2o
from h2o.estimators.kmeans import H2OKMeansEstimator
from h2o.transforms.decomposition import H2OPCA
import logging
import shutil

# Start the logger
logging.config.dictConfig({
    'version':1,
    'disable_existing_loggers': False,
    'handlers':{
        'fileHandler':{
            'class': 'logging.FileHandler',
            'formatter': 'myFormatter',
            'filename': 'app_log.txt'
        }
    },        
    'loggers':{
        '':{
            'handlers': ['fileHandler'],
            'level': 'DEBUG'
        }
    },
    'formatters':{
        'myFormatter':{
            'format': '%(levelname)s - %(asctime)s - Module: %(module)s - %(message)s',
            'datefmt': '%m-%d-%Y %I:%M:%S %p'
        }
    }
})

LOGGER = logging.getLogger(__name__)

########################################
##### Utility functions
########################################

def get_cwd():
    try:
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        LOGGER.info('Successfully mapped to the script directory')
    except:
        os.chdir('/Users/bblanchard006/Desktop/SMU/QTW/Week 5/Summer 2021')
        LOGGER.info('Directory successfully mapped to the local directory')

    active_dir = os.getcwd()
       
    return active_dir

def compile_excel_data(filename, tab_names, new_tab_names, skip_rows = 0):
    
    # Inputs: Excel filename in directory; list of tab names in workbook; None or single string to replace a tab name
    # Description: Confirms an Excel workbook exists (and all required tabs are available); reads in the workbook;
    #              standardizes header names; filters region/market variables; returns a dictionary of dataframes

    master_data = {}
    file_path = 'source_data/{}'.format(filename) 

    for tab in tab_names:
        try:
            if new_tab_names is not None:
                df_name_str = new_tab_names
            else:
                df_name_str = 'ss_'+re.sub(r"\s+", '_', tab)

            LOGGER.info('Extracting the {} data from Excel'.format(df_name_str))

            dframe = pd.read_excel(file_path, tab, skip_rows)

            sanitizer = {
                        '$':'USD',
                        '(':' ',
                        ')':' ',
                        '/':' ',
                        '-':' ',
                        ',':' ',
                        '.':' '
            }
                        
            for key, value in sanitizer.items():
                dframe.rename(columns=lambda x: x.replace(key, value), inplace=True)
                
            dframe.rename(columns=lambda x: x.strip(), inplace=True)
            dframe.rename(columns=lambda x: re.sub(' +','_', x), inplace=True)
            
            dframe.columns = map(str.lower, dframe.columns)
                        
            master_data.update({df_name_str:dframe})
        except Exception as e:
            LOGGER.exception('{} Excel extract failed'.format(df_name_str))
            master_data.update({df_name_str:'Failed'})

        LOGGER.info('{} extraction complete'.format(df_name_str))   
    
    return master_data
    
def compile_csv_data(filename, df_name, prefix):
    
    # Inputs: csv filename in directory

    master_data = {}

    try:
        file_path = 'source_data/{}'.format(filename + '.csv') 

        df_name_str = prefix + '_' + df_name

        LOGGER.info('Extracting the {} data from csv'.format(df_name_str))

        dframe = pd.read_csv(file_path)

        sanitizer = {
                    '$':'USD',
                    '(':' ',
                    ')':' ',
                    '/':' ',
                    '-':' ',
                    ',':' ',
                    '.':' '
        }
                    
        for key, value in sanitizer.items():
            dframe.rename(columns=lambda x: x.replace(key, value), inplace=True)
            
        dframe.rename(columns=lambda x: x.strip(), inplace=True)
        dframe.rename(columns=lambda x: re.sub(' +','_', x), inplace=True)
        
        dframe.columns = map(str.lower, dframe.columns)
        
        master_data.update({df_name_str:dframe})
    except Exception as e:
        LOGGER.exception('{} csv extract failed'.format(df_name_str))
        master_data.update({df_name_str:'Failed'})

    LOGGER.info('{} extraction complete'.format(df_name_str))   
    
    return master_data

def dict_to_csv(mydict, timestamp = False):
    
    # Inputs: a dictionary of dataframes; timestamp = True adds an ISO-formatted suffix to the filename
    # Description: Writes dataframes contained within a dictionary to CSV (on your directory)
    
    folder_path = os.path.join(str(os.getcwd()) + os.sep + 'output')
    suffix = '_' + re.sub(r"\:+", '', dt.datetime.now().isoformat()) + '.csv' if timestamp else '.csv'  
    
    for key, value in mydict.items():
        file_path = os.path.join(folder_path, str(key) + suffix)
        value.to_csv(file_path, sep=',', encoding='utf-8', index = False)     

def drop_cols_by_name(df, list_of_cols):

    for c in list_of_cols:
        del df[c]

    return df        

def get_substring(string, char_len, start_position = 'back'):
    
    if start_position == 'back':
        try:
            sub = string[-char_len:]
        except:
            sub = np.nan
    else:
        try:
            sub = string[:char_len]
        except:
            sub = np.nan
                 
    return sub

def yes_no_to_binary(df, binary_var, no_is_null = True):

    df[binary_var] = df[binary_var].str.lower()
    
    if no_is_null:
        df[binary_var] = np.where(pd.isnull(df[binary_var]), 'n', df[binary_var])
    else:
        pass

    df[binary_var] = df[binary_var].apply(lambda x: get_substring(x, 1, start_position = 'front'))
    
    return df

def yes_no_binary_w_null(df, binary_var, null_label):
    
    df[binary_var] = df[binary_var].str.lower()
    df[binary_var] = df[binary_var].apply(lambda x: get_substring(x, 1, start_position = 'front'))
    df[binary_var] = np.where(pd.isnull(df[binary_var]), null_label, df[binary_var])

    return df

def replace_null_label(df, var, null_label):
    
    df[var] = np.where(pd.isnull(df[var]), null_label, df[var])
    
    return df

def concat_cols_with_sep(df, list_of_cols):
    
    for c in list_of_cols:
        df[c] = df[c].astype(str)
        
    new_col = df[list_of_cols].apply(lambda x: '|'.join(x), axis=1)
    
    return new_col

def find_in_list(label,search_list):
    
    if label in search_list:
        x = 'yes'
    else:
        x = 'no'
    
    return x

def reset_model_training():
    
    dump_folders = [
        'kmeans_models',
        'pca_models'        
    ]
    
    for f in dump_folders:
        temp_delete_path = os.getcwd() + os.sep + f
        shutil.rmtree(temp_delete_path)
        os.mkdir(temp_delete_path)
        
########################################
##### Processing Functions
########################################
        
def get_ratio(df, numerator, denominator, new_col_label):
    
    numerator = pd.to_numeric(df[numerator], errors = 'coerce', downcast = 'float')
    denominator = pd.to_numeric(df[denominator], errors = 'coerce', downcast = 'float')
    
    df[new_col_label] = (numerator / denominator)*100
    
    return df

def create_time_of_day(value):
    
    if value in [22,23,0,1,2,3,4,5]:
        x = 'overnight'
    elif value in [6,7,8,9,10]:
        x = 'morning'
    elif value in [11,12,13,14,15,16]:
        x = 'afternoon'
    elif value in [17,18,19,20,21]:
        x = 'evening'
    else:
        x = 'other'
        
    return x

def drop_null_rows_by_perc(df, threshold):
    
    total_cols = df.shape[1]
    null_cutoff = math.ceil(total_cols*threshold)
    
    df = df.dropna(axis=0, thresh=null_cutoff)
    
    return df

########################################
##### Processing Functions
########################################

def run_w_pca(df, k_val, transform_type, method, id_field):
    
    h2o.init()
    h2o.remove_all()    
    
    i = dt.datetime.now()
    iso=i.strftime('%Y%m%dT%H_%M_%S')

    pca_frame = h2o.H2OFrame.from_python(df)
    pca_model = H2OPCA(
            k = k_val, 
            transform = transform_type, 
            pca_method = method, 
            use_all_factor_levels = True,
            model_id = 'pca_model_' + method + '_' + transform_type + '_k' + str(k_val) + '_' + iso
    )
    
    model_cols = [x for x in pca_frame.columns if id_field not in x]
    
    pca_model.train(x=model_cols, training_frame=pca_frame)
    pca_model_id = pca_model.model_id
    
    pred_frame = pca_model.predict(pca_frame)
    full_frame = pca_frame.cbind(pred_frame)
    full_frame = full_frame.as_data_frame(use_pandas = True)
    
    pca_model_path = os.getcwd() + os.sep + 'pca_models'
    h2o.save_model(pca_model, path = pca_model_path, force=True)

    kmeans_list = [
        {'init_method':'Random', 'standardize_t_f':True},
        {'init_method':'Random', 'standardize_t_f':False},
        # {'init_method':'Furthest', 'standardize_t_f':True},
        # {'init_method':'Furthest', 'standardize_t_f':False},
        # {'init_method':'PlusPlus', 'standardize_t_f':True},
        # {'init_method':'PlusPlus', 'standardize_t_f':False},
    ]

    new_pca_summaries = []

    for index, item in enumerate(kmeans_list):
        for K in range(2,7):
            kmeans_summary = run_kmeans(full_frame, K, item['init_method'], id_field, pca=True, standardize=item['standardize_t_f'])
            kmeans_summary['pca_model_id'] = pca_model_id
            model_summary = pd.DataFrame({'pca_model_id':pca_model_id,'pca_k':k_val,'pca_transform':transform_type,'pca_method':method}, index=[0])
            model_summary = model_summary.merge(kmeans_summary, how='left', on='pca_model_id')
            new_pca_summaries.append(model_summary)

    old_pca_summaries = []

    try:
        pca_model_summaries = pd.read_csv(pca_model_path + os.sep + 'pca_model_summaries.csv')
        old_pca_summaries.append(pca_model_summaries)
    except:
        pass

    if len(old_pca_summaries) > 0:
        new_and_old_pca_summaries = old_pca_summaries + new_pca_summaries
        model_summaries = pd.concat(new_and_old_pca_summaries)
    else:
        model_summaries = pd.concat(new_pca_summaries)
        
    model_summaries.to_csv(pca_model_path+os.sep+'pca_model_summaries.csv', sep=',', encoding='utf-8', index=False)

    return
       
def run_wo_pca(df, id_field):
    
    kmeans_model_path = os.getcwd() + os.sep + 'kmeans_models'
    
    kmeans_list = [
        'Random',
        'Furthest',
        'PlusPlus'
    ]

    new_kmeans_summaries = []

    for init_method in kmeans_list:
        # Change the range (start, stop) to increase/decrease the number of clusters generated
        for K in range(2,7):
            kmeans_summary = run_kmeans(df, K, init_method, id_field, pca=False, standardize=True)
            new_kmeans_summaries.append(kmeans_summary)

    old_kmeans_summaries = []

    try:
        kmeans_model_summaries = pd.read_csv(kmeans_model_path + os.sep + 'kmeans_wo_pca_model_summaries.csv')
        old_kmeans_summaries.append(kmeans_model_summaries)
    except:
        pass

    if len(old_kmeans_summaries) > 0:
        new_and_old_kmeans_summaries = old_kmeans_summaries + new_kmeans_summaries
        model_summaries = pd.concat(new_and_old_kmeans_summaries)
    else:
        model_summaries = pd.concat(new_kmeans_summaries)
        
    model_summaries.to_csv(kmeans_model_path+os.sep+'kmeans_wo_pca_model_summaries.csv', sep=',', encoding='utf-8', index=False)

    return
     
def run_kmeans(df, k_val, init_method, id_field, pca=True, standardize=True):

    h2o.init()
    h2o.remove_all()
    
    i = dt.datetime.now()
    iso=i.strftime('%Y%m%dT%H_%M_%S')
    kmeans_model_id = 'kmeans_model_' + init_method + '_k' + str(k_val) + '_' + iso
    
    if pca:
        base_cols = [id_field]
        pca_cols = [x for x in df.columns if 'PC' in x]
        keep_cols = base_cols + pca_cols
    else:
        keep_cols = df.columns.tolist()

    cluster_df = df[keep_cols]
    cluster_df = h2o.H2OFrame.from_python(cluster_df)    
    cluster_cols = [x for x in keep_cols if id_field not in x]
    
    if standardize:
        kmeans_model = H2OKMeansEstimator(
                k=k_val, 
                init=init_method, 
                seed=2, 
                standardize=True,
                model_id = kmeans_model_id
        )
    else:
        kmeans_model = H2OKMeansEstimator(
                k=k_val, 
                init=init_method, 
                seed=2, 
                standardize=False,
                model_id = kmeans_model_id
        )

    kmeans_model.train(x=cluster_cols, training_frame = cluster_df)
    kmeans_summary = kmeans_model.summary().as_data_frame()
    kmeans_summary = kmeans_summary.drop(kmeans_summary.columns[0], axis=1)
    summary_cols = kmeans_summary.columns.tolist()
    kmeans_summary['model_id'] = kmeans_model_id
    sorted_cols = ['model_id'] + summary_cols
    kmeans_summary = kmeans_summary[sorted_cols]
    
    kmeans_model_path = os.getcwd() + os.sep + 'kmeans_models'
    h2o.save_model(kmeans_model, path = kmeans_model_path, force=True)
    
    return kmeans_summary

def compile_summaries():
    
    kmeans_wo_pca_summaries = []
    
    kmeans_wo_pca_model_path = os.getcwd() + os.sep + 'kmeans_models'

    try:
        kmeans_wo_pca_model_summaries = pd.read_csv(kmeans_wo_pca_model_path + os.sep + 'kmeans_wo_pca_model_summaries.csv')
        kmeans_wo_pca_summaries.append(kmeans_wo_pca_model_summaries)
    except:
        pass

    kmeans_w_pca_summaries = []

    kmeans_w_pca_model_path = os.getcwd() + os.sep + 'pca_models'

    try:
        kmeans_w_pca_model_summaries = pd.read_csv(kmeans_w_pca_model_path + os.sep + 'pca_model_summaries.csv')
        kmeans_w_pca_summaries.append(kmeans_w_pca_model_summaries)
    except:
        pass

    combined_summaries = kmeans_wo_pca_summaries + kmeans_w_pca_summaries

    try:
        combined_df = pd.concat(combined_summaries)
        combined_df = combined_df.sort_values([
                'within_cluster_sum_of_squares',
                'between_cluster_sum_of_squares',
                'total_sum_of_squares'
                ],
            ascending=[
                    1,
                    0,
                    1
            ]
        )

        combined_df.to_csv('complete_model_summaries.csv', index=False)
    except:
        pass
    
    return combined_df
    
def get_best_clusters(df, optional_cluster_value):

    h2o.init()
    h2o.remove_all()
    
    model_summaries = compile_summaries()
    
    model_summaries = model_summaries.sort_values([
                'within_cluster_sum_of_squares',
                'between_cluster_sum_of_squares',
                'total_sum_of_squares'
                ],
            ascending=[
                    1,
                    0,
                    1
            ]
    )
    
    try:    
        best_pca_model_id = model_summaries['pca_model_id']
        best_pca_model_id = best_pca_model_id.iloc[0]
        best_pca_model_path = os.getcwd() + os.sep + 'pca_models' + os.sep + best_pca_model_id
        best_pca_model = h2o.load_model(best_pca_model_path)
        skip_pca = False
    except:
        skip_pca = True

    if optional_cluster_value != None:
        model_summaries = model_summaries[model_summaries['number_of_clusters'] == optional_cluster_value]
        
    best_kmeans_model_id = model_summaries['model_id']
    best_kmeans_model_id = best_kmeans_model_id.iloc[0]
    best_kmeans_model_path = os.getcwd() + os.sep + 'kmeans_models' + os.sep + best_kmeans_model_id
    best_kmeans_model = h2o.load_model(best_kmeans_model_path)
    
    h2o_frame = h2o.H2OFrame.from_python(df)    

    if skip_pca:    
        kmeans_frame = best_kmeans_model.predict(h2o_frame)
        h2o_frame = h2o_frame.cbind(kmeans_frame)
    else:
        pca_frame = best_pca_model.predict(h2o_frame)
        h2o_frame = h2o_frame.cbind(pca_frame)
        kmeans_frame = best_kmeans_model.predict(h2o_frame)
        h2o_frame = h2o_frame.cbind(kmeans_frame)

    python_frame = h2o_frame.as_data_frame(use_pandas=True)
    python_frame['predict'] = python_frame['predict'].apply(lambda x: 'cluster_' + str(x+1))
    python_frame = python_frame.rename(columns={'predict':'predicted_cluster'})
    
    return python_frame

########################################
##### Main Function
########################################

def main(cluster_number_pref = None, id_field = 'encounter_id', train=True, reset_train=False):

    get_cwd()
    
    master_data = {}
    
    csv_files = [
        {'csv':'diabetic_data','name':'raw_data'}
    ]
    
    for index, item in enumerate(csv_files):
        master_data.update(compile_csv_data(item['csv'], item['name'], 'original'))

    # Remove any extra columns containing null values    
    for key, value in master_data.items():
        master_data[key].dropna(axis=1, how='all', inplace=True)

    keep_vars = [
         {'df':'original_raw_data',
              'cols':[
                    'encounter_id',
                    'race',
                    'gender',
                    'age',
                    'weight',
                    'admission_type_id',
                    'time_in_hospital',
                    'payer_code',
                    'medical_specialty',
                    'num_lab_procedures',
                    'num_procedures',
                    'num_medications',
                    'number_outpatient',
                    'number_emergency',
                    'number_inpatient',
                    'number_diagnoses',
                    'max_glu_serum',
                    'a1cresult',
                    'metformin',
                    'repaglinide',
                    'nateglinide',
                    'chlorpropamide',
                    'glimepiride',
                    'acetohexamide',
                    'glipizide',
                    'glyburide',
                    'tolbutamide',
                    'pioglitazone',
                    'rosiglitazone',
                    'acarbose',
                    'miglitol',
                    'troglitazone',
                    'tolazamide',
                    'examide',
                    'citoglipton',
                    'insulin',
                    'diabetesmed',
                    'readmitted'
                      ]},
    ]

    for index, item in enumerate(keep_vars):
        if item['df'] in master_data.keys():
            master_data[item['df']] = master_data[item['df']][item['cols']]

    master_data['modeling_data'] = copy.deepcopy(master_data['original_raw_data'].head(5000))

    if train:
    
        if reset_train:
            reset_model_training()
        
        pca_list = [
            {'transform':'Standardize', 'method':'GLRM'},
            # {'transform':'Normalize', 'method':'GLRM'},
            # {'transform':'Demean', 'method':'GLRM'},
            # {'transform':'Descale', 'method':'GLRM'},
            # {'transform':'None', 'method':'GLRM'},
            # {'transform':'Standardize', 'method':'Randomized'},
            # {'transform':'Normalize', 'method':'Randomized'},
            # {'transform':'Demean', 'method':'Randomized'},
            # {'transform':'Descale', 'method':'Randomized'},
            # {'transform':'None', 'method':'Randomized'},
            # {'transform':'Standardize', 'method':'GramSVD'},
            # {'transform':'Normalize', 'method':'GramSVD'},
            # {'transform':'Demean', 'method':'GramSVD'},
            # {'transform':'Descale', 'method':'GramSVD'},
            # {'transform':'None', 'method':'GramSVD'}
        ]
        
        k_values = []
        # Modify the range (start, stop) below to increase/decrease the number of clusters generated
        for i in range(3,6):
            k_values.append(i)
        
        for index, item in enumerate(pca_list):
            for K in k_values:
                run_w_pca(master_data['modeling_data'], K, item['transform'], item['method'], id_field)
                
        run_wo_pca(master_data['modeling_data'], id_field)
        master_data['clustered_data'] = get_best_clusters(master_data['modeling_data'], cluster_number_pref)        

    else:
        master_data['clustered_data'] = get_best_clusters(master_data['modeling_data'], cluster_number_pref)        

    dict_to_csv(master_data)

    return master_data
            
########################################
##### Main Function
########################################    

if __name__ == "__main__":
    main(cluster_number_pref = None, train=True, reset_train=False)
    pass        





