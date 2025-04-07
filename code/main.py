##################################################################
#
# Project: TME
# Description: Main file to run the project
# Authors: Kushal Virupakshappa
#
# Usage:
#   - python main.py
#
##################################################################

import os
import argparse
import shutil
import json
import pandas as pd
from utils.generic_utils import move_files
from dataset_modules.dataset_tme import PrepareData
from dataset_modules.tokenizer_tme import create_vocab_from_datafile 
from dataset_modules.clinical_data_preprocess import filter_clinical_data
from train import train_wrapper_supervised

def file_utils(src, dest):
    print(f"Moving files from {src} to {dest}")
    move_files(src, dest)



def main(args):
    if args.mode == 'file_utils':
        file_utils(args.src, args.dest)
    elif args.mode == 'filter_clinical':
        # Preprocess clincial data
        filter_clinical_data(args.clinical_data_file_path, args.retain_columns_file_path,args.save_clinical_filtered_path)
        # Create vocabulary from clinical data
    elif args.mode == 'create_vocab':
        vocab,vocab_dtype = create_vocab_from_datafile(args.save_clinical_filtered_path)
        complete_vocabulary = {
            "clinical_features": vocab,
            "clinical_dtype": vocab_dtype
        }
        with open(args.vocabulary_json, 'w') as f:
            json.dump(complete_vocabulary, f)
    elif args.mode == 'prepare_map_data':
        prepare_data = PrepareData(args.filtered_csv_file, args.pathological_file_path, args.pathological_file_extension, args.vocabulary_json)
        prepare_data.save_data(args.save_clinical_filtered_path)
    elif args.mode == 'train':
        # Train the model
        train_wrapper_supervised(args.input_csv_file,args.target_csv_file, args.pathological_file_path, args.pathological_file_extension)
    else:
        raise Exception("Invalid mode of operation")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main file to run the project')
    parser.add_argument('--mode', type=str, help='mode of operation', choices=['file_utils', 'prepare_map_data', 'filter_clinical', 'create_vocab','train'])
    parser.add_argument('--src', type=str, help='source folder path')
    parser.add_argument('--dest', type=str, help='destination folder path')
    parser.add_argument('--filtered_csv_file', type=str, help='CSV file path')
    parser.add_argument('--pathological_file_path', type=str, help='Pathological file path')
    parser.add_argument('--pathological_file_extension', type=str, default='.h5', help='Pathological file extension')
    parser.add_argument('--tokenize_file', type=str, help='File to tokenize')
    parser.add_argument('--clinical_data_file_path', type=str, help='Path to clinical data file')
    parser.add_argument('--retain_columns_file_path', type=str, help='Path to retain columns file')
    parser.add_argument('--save_clinical_filtered_path', type=str, help='Path to save filtered clinical data')
    parser.add_argument('--vocabulary_json', type=str, help='Path to save vocabulary JSON file')
    parser.add_argument('--input_csv_file', type=str, help='Input CSV file path')
    parser.add_argument('--target_csv_file', type=str, help='Target CSV file path')
    args = parser.parse_args()
    main(args)
    