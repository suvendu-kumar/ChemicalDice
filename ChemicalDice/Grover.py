import os
import shutil
import sys
from multiprocessing import Pool
from typing import List, Tuple

from tqdm import tqdm
import subprocess
import pandas as pd
import os
# Arguments to be passed to the called script
from ChemicalDice.utils import get_data, makedirs, load_features, save_features
from ChemicalDice.molfeaturegenerator import get_available_features_generators, \
    get_features_generator
from ChemicalDice.task_labels import rdkit_functional_group_label_features_generator


import random

import numpy as np
import torch
from rdkit import RDLogger
import argparse

from ChemicalDice.parsing import parse_args, get_newest_train_args
from ChemicalDice.utils import create_logger
from ChemicalDice.cross_validate import cross_validate
from ChemicalDice.fingerprint import generate_fingerprints
from ChemicalDice.predict import make_predictions, write_prediction
from ChemicalDice.pretrain import pretrain_model
from ChemicalDice.torchvocab import MolVocab

import requests
from rdkit import Chem

def download_file(url, filename):
    response = requests.get(url, stream=True)
    # Get the total file size
    file_size = int(response.headers.get("Content-Length", 0))
    # Initialize the progress bar
    progress = tqdm(response.iter_content(1024), f"Downloading {filename}", total=file_size, unit="B", unit_scale=True, unit_divisor=1024)
    with open(filename, 'wb') as file:
        for data in progress.iterable:
            # Write data read to the file
            file.write(data)
            # Update the progress bar manually
            progress.update(len(data))


import tarfile

def extract_tar_gz(file_path,path):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path)

def get_grover_prerequisites(path):
    if os.path.exists(os.path.join(path,"grover_large.pt")):
        pass
    else:
        # URL of the file to be downloaded
        url = "https://ai.tencent.com/ailab/ml/ml-data/grover-models/pretrain/grover_large.tar.gz"
        # Name of the file to save as
        filename = os.path.join(path,"grover_large.tar.gz")
        download_file(url, filename)
        print("Grover model is downloaded")
        # Path to the tar.gz file
        extract_tar_gz(filename,path)
        print("Grover model is extracted")
    return os.path.join(path,"grover_large.pt")

    
def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def smile_to_graph(smiles_list, output_dir):
    """
    Computes and saves features for a dataset of molecules as a 2D array in a .npz file.

    :param args: Arguments.
    """
    if not os.path.exists(output_dir):
        print("making directory ", output_dir)
        os.makedirs(output_dir)
    grover_input_file = os.path.join(output_dir,"grover_input.csv")
    with open(grover_input_file, "w") as file1:
        # Writing data to a file
        file1.write("smiles\n")
        input_to_file  = "\n".join(smiles_list)
        file1.writelines(input_to_file)
    features_file = os.path.join(output_dir,"graph_features.npz")
    # Run the called script with arguments
    #subprocess.run(["python", "save_features.py", "--data_path", grover_input_file, "--save_path", features_file , "--features_generator", "rdkit_2d_normalized","--restart"])


    # Create directory for save_path
    makedirs(features_file, isfile=True)

    # Get data and features function
    data = get_data(path=grover_input_file, max_data_size=None)
    features_generator = get_features_generator( "rdkit_2d_normalized")
    temp_save_dir = features_file + '_temp'

    # Load partially complete data
    if True:
        if os.path.exists(features_file):
            os.remove(features_file)
        if os.path.exists(temp_save_dir):
            shutil.rmtree(temp_save_dir)
    else:
        if os.path.exists(features_file):
            raise ValueError(f'"{features_file}" already exists and args.restart is False.')

        if os.path.exists(temp_save_dir):
            features, temp_num = load_temp(temp_save_dir)

    if not os.path.exists(temp_save_dir):
        makedirs(temp_save_dir)
        features, temp_num = [], 0

    # Build features map function
    data = data[len(features):]  # restrict to data for which features have not been computed yet
    mols = (d.smiles for d in data)

    if True:
        features_map = map(features_generator, mols)
    else:
        features_map = Pool(30).imap(features_generator, mols)

    # Get features
    temp_features = []
    for i, feats in tqdm(enumerate(features_map), total=len(data)):
        temp_features.append(feats)

        # Save temporary features every save_frequency
        if (i > 0 and (i + 1) % 10000 == 0) or i == len(data) - 1:
            save_features(os.path.join(temp_save_dir, f'{temp_num}.npz'), temp_features)
            features.extend(temp_features)
            temp_features = []
            temp_num += 1

    try:
        # Save all features
        save_features(features_file, features)

        # Remove temporary features
        shutil.rmtree(temp_save_dir)
    except OverflowError:
        print('Features array is too large to save as a single file. Instead keeping features as a directory of files.')




def load_temp(temp_dir: str) -> Tuple[List[List[float]], int]:
    """
    Loads all features saved as .npz files in load_dir.

    Assumes temporary files are named in order 0.npz, 1.npz, ...

    :param temp_dir: Directory in which temporary .npz files containing features are stored.
    :return: A tuple with a list of molecule features, where each molecule's features is a list of floats,
    and the number of temporary files.
    """
    features = []
    temp_num = 0
    temp_path = os.path.join(temp_dir, f'{temp_num}.npz')

    while os.path.exists(temp_path):
        features.extend(load_features(temp_path))
        temp_num += 1
        temp_path = os.path.join(temp_dir, f'{temp_num}.npz')

    return features, temp_num


def setup(seed):
    # frozen random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



# Get the absolute path of the checkpoint file



def get_embeddings(input_file, output_file_name,output_dir = "temp_data"):
    checkpoint_path_grover = get_grover_prerequisites(output_dir)
    smiles_df = pd.read_csv(input_file)
    if not os.path.exists(output_dir):
        print("making directory ", output_dir)
        os.makedirs(output_dir)
    smiles_list = smiles_df["Canonical_SMILES"]
    if "id" in smiles_df.columns:
        smiles_id_list = smiles_df["id"]
    else:
        smiles_df["id"] = [ "C"+str(id) for id in range(len(smiles_list))]
        smiles_id_list = smiles_df["id"]

    smiles_list_valid = []
    smiles_id_list_valid = []
    for smiles,id in zip(smiles_list,smiles_id_list):
        if is_valid_smiles(smiles):
            smiles_list_valid.append(smiles)
            smiles_id_list_valid.append(id)
        else:
            print("This is a invalid smiles: ", smiles)
    
    smiles_list = smiles_list_valid
    smiles_id_list = smiles_id_list_valid

    smile_to_graph(smiles_list, output_dir)

    grover_input_file = os.path.join(output_dir,"grover_input.csv")
    features_file = os.path.join(output_dir,"graph_features.npz")
    grover_output_model= os.path.join(output_dir,"graph_features.npz")
    smiles_id_list = "___".join(smiles_id_list)
    # Run the called script with arguments
    # subprocess.run(["python", "grovermain.py", "fingerprint", "--data_path", grover_input_file, "--features_path", features_file , "--checkpoint_path", "ckpts/grover_large.pt",
    #             "--fingerprint_source", "both", "--output", grover_output_model, "--dropout", ".2","--grover_output",output_file_name,"--id_list",smiles_id_list])
    

    # Create a namespace object
    args = argparse.Namespace()

    # Now add attributes to the namespace object
    args.data_path = grover_input_file
    args.features_path = features_file
    args.checkpoint_path = checkpoint_path_grover 
    args.fingerprint_source = "both"
    args.output = grover_output_model
    args.dropout = 0.2
    args.grover_output = output_file_name
    args.id_list = smiles_id_list
    args.parser_name = "fingerprint"
    args.output_path = output_dir
    args.no_cuda = True
    args.no_cache = True

    setup(seed=42)
    # Avoid the pylint warning.
    a = MolVocab
    # supress rdkit logger
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    # Initialize MolVocab
    mol_vocab = MolVocab

    args = parse_args(args)
    # Now args is an object that has the same attributes as if you had parsed the command line arguments



    train_args = get_newest_train_args()
    logger = create_logger(name='fingerprint', save_dir=None, quiet=False)

    
    feas = generate_fingerprints(args, logger)
    #np.savez_compressed(args.output_path, fps=feas)

# import subprocess
# grover_input_file = "/storage2/suvendu/chemdice/Benchmark_data_descriptors/chemdice_descriptors/tox21data/graphfiles/grover_input.csv"
# features_file = "/storage2/suvendu/chemdice/Benchmark_data_descriptors/chemdice_descriptors/tox21data/graphfiles/graph_features.npz"
# output_file_name = "a.csv"
# grover_output_model = "fp.npz"
# subprocess.run(["python", "grovermain.py", "fingerprint", "--data_path", grover_input_file, "--features_path", features_file , "--checkpoint_path", "ckpts/grover_large.pt",
#                 "--fingerprint_source", "both", "--output", grover_output_model, "--dropout", ".2","--grover_output",output_file_name])












