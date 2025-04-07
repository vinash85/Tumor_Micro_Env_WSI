######################################################
#
# Authors: Kushal Virupakshappa
# 
# Description: This file contains the utility functions for preprocessing the pathological WSI
#
# Usage:
#   - python test_utils.py
#
######################################################
import os
import shutil

# copy files from subfolders to parent folder
def move_files(src, dest):
    if not os.path.exists(dest):
        os.makedirs(dest)
    if not os.path.exists(src):
        raise Exception("Source folder does not exist")
        
    subfolders = os.listdir(src)
    print("Subfolders: ", subfolders)

    for subfolder in subfolders:
        if os.path.isdir(os.path.join(src, subfolder)):
            files = os.listdir(os.path.join(src, subfolder))
            for file in files:
                if ".svs" in file:
                    shutil.copy(os.path.join(src, subfolder, file), dest)
                    print("Copied file: ", file)

def get_cases(src):
    if not os.path.exists(src):
        raise Exception("Source folder does not exist")
        
    subfolders = os.listdir(src)
    file_dict = {}

    for subfolder in subfolders:
        if os.path.isdir(os.path.join(src, subfolder)):
            files = os.listdir(os.path.join(src, subfolder))
            for file in files:
                if ".svs" in file:
                    file_dict[file] = subfolder
    return file_dict    