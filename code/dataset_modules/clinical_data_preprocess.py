#######################################################################
#
# Project: TME
# Description: Preprocess to required csv format
#######################################################################

import os
import pandas as pd
import numpy as np
import json
# pan_clinical_file_path = "/home/data/all_images/pan_laud_clinical/clinical.project-tcga-paad.2025-03-31/clinical.tsv"
# lung_clinical_file_path = "/home/data/all_images/pan_laud_clinical/clinical.project-tcga-luad.2025-03-31/clinical.tsv"
integer_args = [ 'age_at','days_to']
float_args = []
survival_args = ['survival_2']

def filter_clinical_data(data_file_paths, retain_columns_file_path, save_clinical_filtered_path= None):
    with open(data_file_paths, 'r') as f:
        file_paths = json.load(f)
    clinical_data_file_path = file_paths['clinical_data_file_path']
    followup_data_file_path = file_paths['followup_data_file_path']
    clinical_data = []
    for clinical_file_path in clinical_data_file_path:
        if not os.path.exists(clinical_file_path):
            print(f"File not found: {clinical_file_path}")
            return None
        clinical_data.append(pd.read_csv(clinical_file_path, sep="\t", header=0))
    
    followup_data = []
    for i, followup_file_path in enumerate(followup_data_file_path):
        retain_columns = ["cases.case_id","cases.submitter_id","follow_ups.days_to_follow_up"]
        retain_dict ={"cases.case_id":"case_id","cases.submitter_id":"submitter_id","follow_ups.days_to_follow_up":"days_to_follow_up"}
        if not os.path.exists(followup_file_path):
            print(f"File not found: {followup_file_path}")
            return None
        followup_data.append(pd.read_csv(followup_file_path, sep="\t", header=0))
        # retain the rows of column "follow_ups.timepoint_catogory" with value "Last_Contact"
        followup_data[i] = followup_data[i][followup_data[i]['follow_ups.timepoint_category'] == 'Last Contact']
        # retain the columns
        followup_data[i] = followup_data[i][retain_columns]
        # rename the columns
        followup_data[i].rename(columns=retain_dict, inplace=True)
    

    # Read the retain columns from the file
    with open(retain_columns_file_path, 'r') as f:
        features_to_retain = json.load(f)
    
    retain_columns = features_to_retain['retain_columns']
    retain_dict = features_to_retain['retain_dict']

    for i in range(len(clinical_data)):
        clinical_data[i] = clinical_data[i][retain_columns]
        # Rename the columns
        clinical_data[i].rename(columns=retain_dict, inplace=True)
        # create a new column "days_to_follow_up" and assign the value of "days_to_follow_up" from followup_data
        # where the "case_id" and "submitter_id" are same
        temp_followup = followup_data[i]
        unique_case_ids = temp_followup['case_id'].unique()
        for case_id in unique_case_ids:
            # get the indices of the rows where the case_id is same
            clinical_indices = clinical_data[i][clinical_data[i]['case_id'] == case_id].index
            #find highest index of the case_id
            # if len(clinical_indices) == 0:
            #     continue
            # clinical_index = clinical_indices[-1]
            followup_index = temp_followup[temp_followup['case_id'] == case_id].index
            # get the value of "days_to_follow_up" from followup_data
            days_to_follow_up = temp_followup.loc[followup_index, 'days_to_follow_up'].values[0]
            # assign the value of "days_to_follow_up" to the clinical_data
            clinical_data[i].loc[clinical_indices, 'days_to_follow_up'] = days_to_follow_up


    
    # Combine the DataFrames
    clinical_data = pd.concat(clinical_data, ignore_index=True)
    # replace the values of survival status
    #clinical_data= clinical_data.replace({'survival_2': {'Alive': "'--"}})
    # convert the columns to the required data types
    new_columns = list(retain_dict.values())
    int_columns = []
    for col in integer_args:
        temp_icol = [ column for column in new_columns if col  in column]
        int_columns.extend(temp_icol)
    float_columns = []
    for col in float_args:
        temp_fcol = [ column for column in new_columns if col  in column]
        float_columns.extend(temp_fcol)
    # Convert columns to integer
    clinical_data[int_columns] = clinical_data[int_columns].apply(pd.to_numeric, errors='coerce', downcast='integer')
    # Convert columns to float
    clinical_data[float_columns] = clinical_data[float_columns].apply(pd.to_numeric, errors='coerce', downcast='float')
    # save file path
    if save_clinical_filtered_path is not None:
        clinical_data.to_csv(save_clinical_filtered_path, index=False)
    print(f"Filtered clinical data saved to {save_clinical_filtered_path}")
    return clinical_data



