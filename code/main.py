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
from utils.generic_utils import move_files
from dataset_modules.dataloader_tme import PrepareData
from dataset_modules.tokenizer_tme import create_vocab_from_datafile 
from dataset_modules.clinical_data_preprocess import filter_clinical_data

def file_utils(src, dest):
    print(f"Moving files from {src} to {dest}")
    move_files(src, dest)



def main(args):
    if args.mode == 'file_utils':
        file_utils(args.src, args.dest)
    elif args.mode == 'prepare_data':
        # Preprocess clincial data
        filter_clinical_data(args.clinical_data_file_path, args.retain_columns_file_path,args.save_clinical_filtered_path)
        # Create vocabulary from clinical data
        vocab = create_vocab_from_datafile(args.save_clinical_filtered_path)
        # Save the vocabulary to a JSON file
        with open(args.vocabulary_json, 'w') as f:
            json.dump(vocab, f)
        #prepare_data = PrepareData(args.csv_file, args.pathological_file_path, args.pathological_file_extension)
        # prepare_data.tokenize_data(args.tokenize_file)
    else:
        raise Exception("Invalid mode of operation")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main file to run the project')
    parser.add_argument('--mode', type=str, help='mode of operation', choices=['file_utils', 'prepare_data'])
    parser.add_argument('--src', type=str, help='source folder path')
    parser.add_argument('--dest', type=str, help='destination folder path')
    parser.add_argument('--csv_file', type=str, help='CSV file path')
    parser.add_argument('--pathological_file_path', type=str, help='Pathological file path')
    parser.add_argument('--pathological_file_extension', type=str, default='.h5', help='Pathological file extension')
    parser.add_argument('--tokenize_file', type=str, help='File to tokenize')
    parser.add_argument('--clinical_data_file_path', type=str, help='Path to clinical data file')
    parser.add_argument('--retain_columns_file_path', type=str, help='Path to retain columns file')
    parser.add_argument('--save_clinical_filtered_path', type=str, help='Path to save filtered clinical data')
    parser.add_argument('--vocabulary_json', type=str, help='Path to save vocabulary JSON file')
    args = parser.parse_args()
    main(args)
    