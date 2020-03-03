
import os, sys
import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import csv, re


def read_txt_files(path="datasets/training"):
    datasets = []
    flist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                flist.append(os.path.join(root, file))
    meta_file_path = os.path.join(path,'imfdb_meta.csv')
    with open(meta_file_path, mode='w') as csv_file:
        fieldnames = ['path','gender','emotion','obstruction','illumination','orientation']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for f in tqdm(flist):
            with open(f) as txt:
                x = list(filter(None, txt.read().split('\n')))
                for xs in x:
                    line = np.array(re.split("\t|\s",xs))
                    if len(line)>16 :
                        file_path = "/".join(f.split('/')[:-1]) + "/images/" + line[2]
                        row_dict = {fieldnames[0]:file_path,fieldnames[1]:line[10],fieldnames[2]:line[11],fieldnames[3]:line[12],fieldnames[4]:line[13],fieldnames[5]:line[15]}
                        writer.writerow(row_dict)

    return meta_file_path

meta_data_path = read_txt_files(path = "/mnt/InternalStorage/sidkas/FastFace/src/datasets/IMFDB_final")

df = pd.read_csv(meta_data_path)


df.describe()

print(df.gender.unique(),df.emotion.unique(),df.obstruction.unique(),df.illumination.unique(),df.orientation.unique())


df = df[df.groupby('gender')['path'].nunique()>500]


df.groupby('emotion')['path'].nunique()>500


