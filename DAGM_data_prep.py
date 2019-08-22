#!/usr/bin/env python
# coding: utf-8


import os
import shutil
import csv
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("datasetdir")
parser.add_argument("dagmfolder")
args = parser.parse_args()


datasetdir = args.datasetdir # replace this with the dataset directory
dagmfolder = args.dagmfolder # replace this with the dagm folder


if not os.path.isdir(datasetdir):
    os.mkdir(datasetdir)

if not os.path.isdir(dagmfolder):
    raise ValueError('Not a Valid Path')

os.chdir(datasetdir)
os.mkdir('./DAGM_classification')
os.mkdir('./DAGM_segmentation')




os.chdir(os.path.join(datasetdir,'DAGM_classification'))
for i in range(1,11):
    os.mkdir(f'./Class{i}')
    for subset in ['Train','Test']:
        os.mkdir(f'./Class{i}/{subset}')
        for category in ['Defect','NonDefect']:
            os.mkdir(f'./Class{i}/{subset}/{category}')          




for i in range(1,11):
    print(f'Class {i}')
    for subset in ['Train','Test']:
        with open(os.path.join(dagmfolder,f'Class{i}/{subset}/Label/Labels.txt'), newline='\n') as csvfile:
            filereader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            next(filereader,None)
            for row in filereader:
                if int(row[1])==1:
                    shutil.copy(os.path.join(dagmfolder,f'Class{i}/{subset}/{row[2]}'),f'./Class{i}/{subset}/Defect')
                else:
                    shutil.copy(os.path.join(dagmfolder,f'Class{i}/{subset}/{row[2]}'),f'./Class{i}/{subset}/NonDefect')




for i in range(1,11):
    for subset in ['Train','Test']:
        for category in ['Defect','NonDefect']:
            val = len(os.listdir(os.path.join(datasetdir,f'DAGM_classification/Class{i}/{subset}/{category}')))
            print(f'Class{i} {subset} {category}: {val}')




os.chdir(os.path.join(datasetdir,'DAGM_segmentation'))
for i in range(1,11):
    os.mkdir(f'./Class{i}/')
    for subset in ['Train','Test']:
        os.mkdir(f'./Class{i}/{subset}')
        for fol in ['Image','Mask']:
            os.mkdir(f'./Class{i}/{subset}/{fol}')




for i in range(1,11):
    print(f'Class {i}')
    for subset in ['Train','Test']:
        print(f'Copying {subset}')
        with open(os.path.join(dagmfolder,f'Class{i}/{subset}/Label/Labels.txt'), newline='\n') as csvfile:
            filereader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            next(filereader,None)
            for row in filereader:
                shutil.copy(os.path.join(dagmfolder,f'Class{i}/{subset}/{row[2]}'),f'./Class{i}/{subset}/Image')
                if int(row[1])==1: #defect
                    shutil.copy(os.path.join(dagmfolder,f'Class{i}/{subset}/Label/{row[4]}'),f'./Class{i}/{subset}/Mask')
                else:
                    cv2.imwrite(f'./Class{i}/{subset}/Mask/{row[0]}_label.PNG',np.zeros((512,512),dtype=np.uint8))




for i in range(1,11):
    for subset in ['Train','Test']:
        for fol in ['Image','Mask']:
            val = len(os.listdir(os.path.join(datasetdir,f'DAGM_segmentation/Class{i}/{subset}/{fol}')))
            print(f'Class{i} {subset} {fol}: {val}')

