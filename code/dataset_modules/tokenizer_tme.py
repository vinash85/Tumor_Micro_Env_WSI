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
import math

def get_vocabulary_from_json(json_file):
    """
    Load vocabulary from a JSON file.
    """
    with open(json_file, 'r') as f:
        vocab = json.load(f)
    clinical_features = vocab['clinical_features']
    clinical_dtype = vocab['clinical_dtype']
    return clinical_features, clinical_dtype

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
    unique_dict={}
    if len(unique_values) == 0:
        raise ValueError("No unique values found.")
    d_type = check_dtype(unique_values)
    if d_type == "string":
        unique_dict= {value: i+1 for i, value in enumerate(unique_values)}
    elif d_type == "integer":
            unique_dict = {}
    elif d_type == "float":
            unique_dict = {}
    return unique_dict, d_type


def create_vocab_from_datafile(data_file):
    """
    Create a vocabulary from a DataFrame column.
    """
    # implementation of creating vocabulary from clinical data
    
    vocab = {}
    vocab_dtype = {}
    df = pd.read_csv(data_file)
    clinical_features = df.columns.tolist()
    for clinical_feature in clinical_features:
        if clinical_feature == "case_id" or clinical_feature == "submitter_id" or clinical_feature == "organ":
            continue
        # Get unique values for the clinical feature
        unique_values = df[clinical_feature].unique()
        unique_dict, d_type = create_unique_dict(unique_values)
        vocab[clinical_feature] = unique_dict
        vocab_dtype[clinical_feature] = d_type
    return vocab , vocab_dtype

class TokenizerTME:
    def __init__(self,clinical_features,vocab_file):
        """
        Initialize the TokenizerTME class.
        """
        self.vocab_file = vocab_file
        self.clinical_features_dict, self.clinical_dtype_dict = get_vocabulary_from_json(self.vocab_file)

        print(f"Vocabulary loaded from JSON file: {self.vocab_file}.")
        self.vocab_keys = list(self.clinical_features_dict.keys())
        if check_vocabulary_features(self.vocab_keys, clinical_features):
            print("All keys are present in the features.")
        else:
            raise ValueError("Some keys are missing in the features.")
        self.clinical_features = clinical_features
        self.assign_col = None
    
        
    def tokenize(self, data_file,save_path=None):
        """
        Tokenize the input data.
        """
        df = pd.read_csv(data_file)
        # tokenize the dataframe
        for clinical_feature in self.clinical_features:
            if clinical_feature == "case_id" or clinical_feature == "submitter_id" or clinical_feature == "organ":
                continue
            if clinical_feature not in df.columns:
                raise ValueError(f"Clinical feature {clinical_feature} not found in the data file.")
            if self.clinical_dtype_dict[clinical_feature] == "string":
                df = df.replace({clinical_feature: self.clinical_features_dict[clinical_feature]})
            elif self.clinical_dtype_dict[clinical_feature] == "integer":
                df[clinical_feature] = df[clinical_feature].astype(int)
            elif self.clinical_dtype_dict[clinical_feature] == "float":
                df[clinical_feature] = df[clinical_feature].astype(float)
            else:
                raise ValueError(f"Unsupported datatype: {self.clinical_dtype_dict[clinical_feature]}")
        return df
        
            
            
