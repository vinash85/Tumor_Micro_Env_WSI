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
integer_args = [ 'age_at']
float_args = []

def filter_clinical_data(clinical_data_file_path, retain_columns_file_path, save_clinical_filtered_path= None):
    with open(clinical_data_file_path, 'r') as f:
        clinical_data_file_path = json.load(f)

    clinical_data = []
    for clinical_file_path in clinical_data_file_path:
        if not os.path.exists(clinical_file_path):
            print(f"File not found: {clinical_file_path}")
            return None
        clinical_data.append(pd.read_csv(clinical_file_path, sep="\t", header=0))
    
    # Read the retain columns from the file
    with open(retain_columns_file_path, 'r') as f:
        features_to_retain = json.load(f)
    
    retain_columns = features_to_retain['retain_columns']
    retain_dict = features_to_retain['retain_dict']

    for i in range(len(clinical_data)):
        clinical_data[i] = clinical_data[i][retain_columns]
        # Rename the columns
        clinical_data[i].rename(columns=retain_dict, inplace=True)
    
    # Combine the DataFrames
    clinical_data = pd.concat(clinical_data, ignore_index=True)
    # convert the columns to the required data types
    new_columns = list(retain_dict.values())
    int_columns = []
    for col in integer_args:
        temp_col = [ column for column in new_columns if col  in column]
        int_columns.extend(temp_col)
    float_columns = []
    for col in float_args:
        temp_col = [ column for column in new_columns if col  in column]
        float_columns.extend(temp_col)
    # Convert columns to integer
    clinical_data[int_columns] = clinical_data[int_columns].apply(pd.to_numeric, errors='coerce', downcast='integer')
    # Convert columns to float
    clinical_data[float_columns] = clinical_data[float_columns].apply(pd.to_numeric, errors='coerce', downcast='float')
    # save file path
    if save_clinical_filtered_path is not None:
        clinical_data.to_csv(save_clinical_filtered_path, index=False)
    print(f"Filtered clinical data saved to {save_clinical_filtered_path}")
    return clinical_data





# retain_columns = [
#     "project.project_id",
#     "cases.case_id",
#     "cases.disease_type",
#     "cases.lost_to_followup",
#     "cases.primary_site",
#     "cases.submitter_id",
#     "demographic.age_at_index",
#     "demographic.race",
#     "demographic.gender",
#     "demographic.vital_status",
#     "diagnoses.age_at_diagnosis",
#     "diagnoses.ajcc_pathologic_stage",
#     "diagnoses.ajcc_pathologic_t",
#     "diagnoses.ajcc_staging_system_edition",
#     "diagnoses.classification_of_tumor",
#     "diagnoses.diagnosis_is_primary_disease",
#     "diagnoses.primary_diagnosis",
#     "diagnoses.site_of_resection_or_biopsy",
#     "diagnoses.sites_of_involvement",
#     "diagnoses.tissue_or_organ_of_origin",
#     "treatments.therapeutic_agents",
#     "treatments.treatment_or_therapy",
#     "treatments.treatment_type"
# ]
# #"treatments.treatment_outcome": "treatment_outcome",
# #    "diagnoses.prior_malignancy": "prior_malignancy",
# #    "diagnoses.prior_treatment": "prior_treatement",
# items_dict = {
#     "project.project_id": "organ",
#     "cases.case_id": "case_id",
#     "cases.disease_type": "disease_type",
#     "cases.lost_to_followup": "survival_1",
#     "cases.primary_site": "primary_site",
#     "cases.submitter_id": "submitter_id",
#     "demographic.age_at_index": "age_at_index",
#     "demographic.race": "race",
#     "demographic.gender": "gender",
#     "demographic.vital_status": "survival_2",
#     "diagnoses.age_at_diagnosis": "age_at_diagnosis",
#     "diagnoses.ajcc_pathologic_stage": "pathologic_stage",
#     "diagnoses.ajcc_pathologic_t": "pathologic_t",
#     "diagnoses.ajcc_staging_system_edition": "staging",
#     "diagnoses.classification_of_tumor": "tumor_class",
#     "diagnoses.diagnosis_is_primary_disease": "primary_disease",
#     "diagnoses.primary_diagnosis": "primary_diagnosis",
#     "diagnoses.site_of_resection_or_biopsy": "site_biopsy",
#     "diagnoses.sites_of_involvement": "specific_site",
#     "diagnoses.tissue_or_organ_of_origin": "tissue_organ_origin",
#     "treatments.therapeutic_agents": "theraputic_agents",
#     "treatments.treatment_or_therapy": "treatment_or_therapy",
#     "treatments.treatment_type": "treatement_type"
# }