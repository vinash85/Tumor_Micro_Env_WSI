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
    def __init__(self, filtered_clinical_csv_file, pathological_file_path, pathological_file_extension=".h5"):
        self.data = pd.read_csv(filtered_clinical_csv_file)
        self.csv_file = filtered_clinical_csv_file
        self.patient_ids = self.data['case_id'].values
        self.submitter_ids = self.data['submitter_id'].values
        self.tokenized_data = self.tokenize_data(self.csv_file)
        self.pathological_data = self.get_pathological_data(self.patient_ids, pathological_file_path=pathological_file_path,
                                                            pathological_file_extension=pathological_file_extension)
        self.survival_data = self.get_survival_data()
        self.clinical_data = self.data.drop(columns =['survival_status'])
    
    def tokenize_data(self, tokenize_file):
        """
        Tokenize the clinical data.
        """
        tokenizer = TokenizeTME()
        tokenized_data = tokenizer.tokenize(tokenize_file)
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
    def get_pathological_data(self, patient_ids, pathological_file, pathological_file_extension=".h5"):
        """
        Retrieve pathological data for the given patient IDs.
        """
        pathological_files = os.listdir(pathological_file_path)
        pathological_features = []
        for uuid in patient_ids:
            temp_file = [f for f in pathological_files if uuid in f]
            if len(temp_file) == 0:
                raise ValueError(f"Pathological file for {uuid} not found.")
            if len(temp_file) > 1:
                raise ValueError(f"Multiple pathological files found for {uuid}.")
            else:
                pathological_file = os.path.join(pathological_file_path, temp_file[0])
                if pathological_file_extension == ".h5":
                    pathological_data_content = h5py.File(pathological_file, 'r')
                    pathological_features [uuid]= pathological_data_content['features'][:]
                    pathological_features [uuid]= flatten(pathological_features[uuid])
                else:
                    raise ValueError(f"Unsupported file extension: {pathological_file_extension}")
        
        return pathological_features
    
    def get_survival_data(self):
        """
        Retrieve survival data from the CSV file.
        """
        survival_data = self.data[['UUID', 'survival_status']]
        return survival_data
    
    def save_data(self, save_path):
        """
        Save the processed data to a CSV file.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.tokenized_data.to_csv(os.path.join(save_path, 'tokenized_data.csv'), index=False)
        self.survival_data.to_csv(os.path.join(save_path, 'survival_data.csv'), index=False)
