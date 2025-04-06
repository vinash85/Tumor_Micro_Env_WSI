#######################################################
#
# This file is part of the TME project.
#
# Author: Kushal Virupakshappa
#
# Date: 2023-10-01 
#
# Description: Tokenize Clinical data and flatten pathological data
#
#########################################################

import os
import json
import pandas as pd
import numpy as np
import torch


def get_vocabulary_from_json(json_file):
    """
    Load vocabulary from a JSON file.
    """
    with open(json_file, 'r') as f:
        vocab = json.load(f)
    return vocab

def check_vocabulary_features(vocab_keys, features):
    """
    Check if the vocabulary keys are present in the features.
    """
    for key in vocab_keys:
        if key not in features:
            raise ValueError(f"Key {key} not found in features.")
    return True

def check_dtype(unique_values):
    """
    Check if the unique values are specific datatypes.
    """
    str_count = 0
    int_count = 0
    float_count = 0
    for value in unique_values:
        if isinstance(value, str):
            str_count += 1
        elif isinstance(value, int):
            int_count += 1
        elif isinstance(value, float):
            float_count += 1
        elif isinstance(value, np.int64):
            int_count += 1
        elif isinstance(value, np.float64):
            float_count += 1
        elif isinstance(value, NoneType):
            continue
        else:
            raise ValueError(f"Unsupported datatype: {type(value)}")
    if str_count > 0 and int_count == 0 and float_count == 0:
        return "string"
    elif str_count == 0 and int_count > 0 and float_count == 0:
        return "integer"
    elif str_count == 0 and int_count == 0 and float_count > 0:
        return "float"


def create_unique_dict(unique_values):
    """
    Create a unique dictionary from unique values.
    """
    unique_dict = {}
    if len(unique_values) == 0:
        raise ValueError("No unique values found.")
    d_type = check_dtype(unique_values)
    if d_type == "string":
        unique_dict = {value: i+1 for i, value in enumerate(unique_values)}
    elif d_type == "integer":
        if len(unique_values) < 10:
            unique_dict = { i+1:value for i, value in enumerate(unique_values) if value != np.nan}
            unique_dict [len(unique_values) + 2] = np.nan
        elif len(unique_values) <= 50:
            #find min and max excluding nan
            x_min = np.nanmin(unique_values)
            x_max = np.nanmax(unique_values)
            bin_size = int((x_max - x_min)//len(unique_values))
            unique_dict = {i+1:[x_min + (i) * bin_size , x_min+ (i+1)*bin_size]  for i in range(len(unique_values))}
            unique_dict [51] = np.nan
        else:
            x_min = np.nanmin(unique_values)
            x_max = np.nanmax(unique_values)
            bin_size = int((x_max - x_min)//50)
            unique_dict = {i+1:[x_min + (i) * bin_size , x_min+ (i+1)*bin_size]  for i in range(50)}
            unique_dict [51] = np.nan
    elif d_type == "float":
            x_min = np.nanmin(unique_values)
            x_max = np.nanmax(unique_values)
            bin_size = int((x_max - x_min)//50)
            x_min = int(x_min)
            unique_dict = {i+1:[x_min + (i) * bin_size , x_min+ (i+1)*bin_size] for i in range(50) }
            unique_dict [51] = np.nan
    return unique_dict


def create_vocab_from_datafile(data_file):
    """
    Create a vocabulary from a DataFrame column.
    """
    # implementation of creating vocabulary from clinical data
    
    vocab = {}
    df = pd.read_csv(data_file)
    clinical_features = df.columns.tolist()
    for clinical_feature in clinical_features:
        # Get unique values for the clinical feature
        unique_values = df[clinical_feature].unique()
        unique_dict = create_unique_dict(unique_values)
        vocab[clinical_feature] = unique_dict
    return vocab

class TokenizerTME:
    def __init__(self,clinical_features,vocab_file):
        """
        Initialize the TokenizerTME class.
        """
        self.vocab_file = vocab_file
        self.vocab_dtype = get_vocabulary_from_json(self.vocab_dtype_file)
        self.complete_vocabulary = get_vocabulary_from_json(self.vocab_file)

        print(f"Vocabulary loaded from JSON file: {self.vocab_file}.")
        self.vocab_keys = list(self.complete_vocabulary.keys())
        if check_vocabulary_features(self.vocab_keys, clinical_features):
            print("All keys are present in the features.")
        else:
            raise ValueError("Some keys are missing in the features.")
        self.clinical_features = clinical_features
    
    def tokenize(self, data_file,save_path=None):
        """
        Tokenize the input data.
        """
        df = pd.read_csv(data_file)
        # each row is a patient and each column is a clinical feature
        # convert each patient's clinical feature into token ids
        for clinical_feature in self.clinical_features:
            if clinical_feature not in df.columns:
                raise ValueError(f"Clinical feature {clinical_feature} not found in the data file.")
            # Tokenize the clinical feature
            df[clinical_feature] = df[clinical_feature].replace(self.complete_vocabulary[clinical_feature])
            #save the tokenized data
        df.to_csv(save_path, index=False)
        return df
            
            
