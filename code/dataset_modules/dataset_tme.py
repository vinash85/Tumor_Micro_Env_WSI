########################################################
"""

This module is responsible for handling data loading operations for the TME project. 
It provides functionality to load, preprocess, and manage datasets efficiently.

Author: Kushal Virupakshappa
Date: 2023-10-04
"""
########################################################


import os
import pandas as pd
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MaxAbsScaler
from .tokenizer_tme import TokenizerTME

class PrepareData:
    """
    tokenize clinical data and flatten pathology features.
    """
    def __init__(self, filtered_clinical_csv_file, pathological_file_path, pathological_file_extension=".h5",vocab_file=None, flatten_dim=False):
        self.flatten_dim = flatten_dim
        self.data = pd.read_csv(filtered_clinical_csv_file)
        self.vocab_file = vocab_file
        self.csv_file = filtered_clinical_csv_file
        self.clinical_features = self.data.columns.tolist()

        self.tokenized_data = self.tokenize(self.csv_file)
        #remove column case_id organ
        columns_to_remove = ['case_id', 'organ']
        self.tokenized_data = self.tokenized_data.drop(columns=columns_to_remove)

        #remove duplicates with same submitter_id
        self.tokenized_data = self.tokenized_data.drop_duplicates(subset=['submitter_id'])
        #remove rows with value 3 from survival_2
        self.tokenized_data = self.tokenized_data[self.tokenized_data['survival_2'] != 3]
        
        print(f'samples of tokenized_data:{self.tokenized_data.head()}')
        self.patient_ids = self.data['submitter_id'].unique().tolist()
        print(f"Number of patients: {len(self.patient_ids)}")
        

        #self.get_pathological_data(self.patient_ids, pathological_file_path=pathological_file_path,
        #                                                    pathological_file_extension=pathological_file_extension)
        self.survival_data, self.tokenized_clinical_data = self.get_survival_data()
        
    
    def tokenize(self, tokenize_file):
        """
        Tokenize the clinical data.
        """
        tokenizer = TokenizerTME(self.clinical_features, self.vocab_file)
        tokenized_data = tokenizer.tokenize(self.csv_file)
        return tokenized_data
    
    def flatten(self, data):
        """
        Flatten the data if it is a nested structure.
        """
        flattened_data = []
        for data_item in data:
            data_item = np.array(data_item)
            if data_item.ndim > 1:
                flattened_data.append(data_item.flatten())
        return flattened_data

    def get_pathological_data(self, patient_ids, pathological_file_path, pathological_file_extension=".h5"):
        """
        Retrieve pathological data for the given patient IDs.
        """
        pathological_files = os.listdir(pathological_file_path)
        print(f"Number of pathological files: {len(pathological_files)}")
        #pathological_features = {}
        no_pathological_files = []
        for uuid in patient_ids:
            temp_file = [f for f in pathological_files if uuid in f]
            if len(temp_file) == 0:
                no_pathological_files.append(uuid)
                # remove the row from the tokenized data
                self.tokenized_data = self.tokenized_data[self.tokenized_data['submitter_id'] != uuid]
            else:
                print(f"no of files found for {uuid}: {len(temp_file)}")   
                # Process all files for the patient
                features = []
                for file in temp_file:
                    print(f"Processing file: {file}")
                    pathological_file = os.path.join(pathological_file_path, file)
                    
                    if pathological_file_extension == "h5":
                        with h5py.File(pathological_file, 'r') as f:
                            temp_features = f['features'][:]
                            if self.flatten_dim:
                                temp_features = self.flatten(temp_features)
                            features.append(temp_features)
                    else:
                        raise ValueError(f"Unsupported file extension: {pathological_file_extension}") 

                if not os.path.exists(os.path.join(pathological_file_path,'filtered')): 
                    os.makedirs(os.path.join(pathological_file_path,'filtered'))

                save_path_file = os.path.join(pathological_file_path,'filtered' ,'collated_'+uuid+'.h5')          
                with h5py.File(save_path_file, 'a') as f:
                    for i in range(len(features)):
                        dset = f.create_dataset(f'data_{i}', features[i].shape, dtype=features[i].dtype)
                        dset[:] = features[i]

    
    def get_survival_data(self):
        """
        Retrieve survival data from the CSV file.
        """
        survival_data = self.tokenized_data[['submitter_id', 'survival_2', 'survival_1','days_to_death', 'days_to_follow_up']]
        # columns other than survival_2, survival_1, days_to_death, days_to_followup
        clinical_data = self.tokenized_data.drop(columns=[ 'survival_2', 'survival_1','days_to_death', 'days_to_follow_up'])
        # remove rows with value 3 from survival_2
        survival_data = survival_data[survival_data['survival_2'] != 3]
        # remove rows with value NaN from age_at_index
        clinical_data = clinical_data[clinical_data['age_at_index'].notna()]
        # remove rows with value NaN from days_to_follow_up
        survival_data = survival_data[survival_data['days_to_follow_up'].notna()]
        return survival_data, clinical_data
    
    def save_data(self, save_path):
        """
        Save the processed data to a CSV file.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.tokenized_data.to_csv(os.path.join(save_path, 'tokenized_complete_data.csv'), index=False)
        self.survival_data.to_csv(os.path.join(save_path, 'tokenized_survival_data.csv'), index=False)
        self.tokenized_clinical_data.to_csv(os.path.join(save_path, 'tokenized_clinical_data.csv'), index=False)


class TMEDataSet(Dataset):
    """
    Dataset class for TME data.
    """
    def __init__(self, input_file,target_file, pathological_file_path, pathological_file_extension=".h5",random_seed=42):
        """
        Initialize the dataset with the CSV file and pathological file path.
        """
        super().__init__()
        self.tokenized_data = pd.read_csv(input_file)
        self.patient_ids = self.tokenized_data['submitter_id'].unique().tolist()
        self.pathological_file_path = pathological_file_path
        self.pathological_file_extension = pathological_file_extension
        self.pathology_files = [ os.path.join(self.pathological_file_path, 'filtered','collated_'+pid+pathological_file_extension) for pid in self.patient_ids]
        self.check_pathology_files()
        self.target_data = pd.read_csv(target_file)
        self.rng = np.random.default_rng(random_seed)
    
    def check_pathology_files(self):
        for file in self.pathology_files:
            if not os.path.exists(file):
                raise Exception(f"Pathology file {file} does not exist")
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.patient_ids)
    
    def read_h5py(self, file):
        """
        Read the HDF5 file and return the data.
        """

        with h5py.File(file, 'r') as f:
            data = []
            for key in f.keys():
                data.append(f[key][:])
        select_key_idx = np.random.choice(len(data), 1, replace=False)
        selected_data = data[select_key_idx[0]]
        # randomize the 2d data
        selected_data = np.random.permutation(selected_data)
        # select first 500 rows
        selected_data = selected_data[:5]
        return selected_data

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        pathology_file = None
        for file in self.pathology_files:
            if patient_id in file:
                pathology_file = file
                break
        input_data = self.tokenized_data[self.tokenized_data['submitter_id'] == patient_id].drop(columns=['submitter_id'])
        input_data = input_data.iloc[0].to_dict()
        input_data['pathology_patches'] = self.read_h5py(pathology_file)
        target_data = self.target_data[self.target_data['submitter_id'] == patient_id]
        target_data = target_data.iloc[0].to_dict()
        inp = [int(input_data['disease_type']),
                int(input_data['primary_site']),
                input_data['age_at_index'],
                int(input_data['race']),
                int(input_data['gender']),
                #input_data['age_at_diagnosis'],
                #input_data['days_to_birth'],
                #int(input_data['pathologic_stage']),
                #int(input_data['staging']),
                #int(input_data['tumor_class']),
                #int(input_data['primary_disease']),
                #int(input_data['primary_diagnosis']),
                #int(input_data['site_biopsy']),
                #int(input_data['specific_site']),
                #int(input_data['tissue_organ_origin']),
                #int(input_data['theraputic_agents']),
                #int(input_data['treatment_or_therapy']),
                #int(input_data['treatement_type']),
            ]
        tar_event = 1 if target_data['survival_2'] == 1 else 0
        tar = [tar_event, target_data['days_to_follow_up']]
        # Convert to tensors
        inp = [torch.tensor([i]) for i in inp]
        tar = [torch.tensor([i]) for i in tar]
        # append the pathology patches
        inp.append(torch.tensor(input_data['pathology_patches']))
        return inp, tar

        
        


