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
from utils import move_files


def file_utils(src, dest):
    move_files(src, dest)



def main(args):
    if args.mode == 'file_utils':
        file_utils(args.src, args.dest)
    else:
        raise Exception("Invalid mode of operation")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main file to run the project')
    parser.add_argument('--mode', type=str, help='mode of operation', choices=['file_utils'])
    parser.add_argument('--src', type=str, help='source folder path')
    parser.add_argument('--dest', type=str, help='destination folder path')
    args = parser.parse_args()
    main(args)
    