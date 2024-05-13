
import os
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from ChemicalDice.image_dataloader import ImageDataset2, load_filenames_and_labels_multitask2, get_datasets2
from ChemicalDice.cnn_model_utils import load_model, train_one_epoch_multitask, evaluate_on_multitask, save_finetune_ckpt
from ChemicalDice.train_utils import fix_train_random_seed, load_smiles
from ChemicalDice.public_utils import cal_torch_model_params, setup_device, is_left_better_right
from ChemicalDice.splitter import split_train_val_test_idx, split_train_val_test_idx_stratified, scaffold_split_train_val_test, \
    random_scaffold_split_train_val_test, scaffold_split_balanced_train_val_test

import requests

def download_file(url, filename):
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)

def get_imagemol_prerequisites(path):
    # URL of the file to be downloaded
    url = "https://raw.githubusercontent.com/suvendu-kumar/ImageMol_model/main/ImageMol.pth.tar"
    # Name of the file to save as
    filename = os.path.join(path,"ImageMol.pth.tar")
    download_file(url, filename)
    print("ImageMol model is downloaded")
    return filename

#import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import pandas as pd
import csv


def writeEmbeddingsIntoFile(file_path, file_name, ids, embeddings):
    if not os.path.exists(file_path):
        os.makedirs(file_path) 
    with open(file_path + file_name, mode='w', newline='') as file:
        writer = csv.writer(file) 
        for row in range(len(ids)):
            writer.writerow(np.concatenate(([ids[row]], embeddings[row])))
    print(len(ids), embeddings.shape)

def get_filename_without_extension(file_path):
    base_name = os.path.basename(file_path)
    filename_without_extension = os.path.splitext(base_name)[0]
    return filename_without_extension

from rdkit import Chem
from rdkit.Chem import Draw

def smile_to_img(smiles_list,smiles_id_list,output_dir):
    image_file_paths = []
    for smis, index in zip(smiles_list,smiles_id_list):
        file_path = os.path.join(output_dir,index + ".png")
        #image = Smiles2Img2(smis, , savePath=file_path)
        size=224
        mol = Chem.MolFromSmiles(smis)
        img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(size, size),returnPNG=False)
        if file_path is not None:
            img.save(file_path)
        if img is None:
            print("Error in smile to image for ", index, smis)
            image_file_paths.append('')
        else:    
            image_file_paths.append(file_path)
    return image_file_paths


def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def add_image_files(input_file, output_dir):
    smiles_df = pd.read_csv(input_file)
    if not os.path.exists(output_dir):
        print("making directory ", output_dir)
        os.makedirs(output_dir)
    smiles_list = smiles_df['Canonical_SMILES']
    if 'id' in smiles_df.columns:
        smiles_id_list = smiles_df['id']
    else:
        smiles_df['id'] = [ "C"+str(id) for id in range(len(smiles_list))]
        smiles_id_list = smiles_df['id']
    
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

    image_file_paths = smile_to_img(smiles_list, smiles_id_list, output_dir)
    smiles_df['image_files'] = image_file_paths
    smiles_df.to_csv(input_file,index=False)

import pkg_resources

# Get the absolute path of the checkpoint file


def image_to_embeddings(input_file, output_file_name):
    #csv_filename = input_file
    #image_folder_224 = input_dir
    add_image_files(input_file, output_dir = "temp_data/images/")
    checkpoint_path = get_imagemol_prerequisites("temp_data/")
    resume = checkpoint_path
    image_model = "ResNet18"
    imageSize = 224
    ngpu = 1
    runseed=2021
    workers = 2

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #image_folder, txt_file = get_datasets2(dataset=csv_filename, dataroot=image_folder_224, data_type="processed")
    verbose = False

    device, device_ids = setup_device(ngpu)

    # fix random seeds
    fix_train_random_seed()

    # architecture name
    if verbose:
        print('Architecture: {}'.format(image_model))
    num_tasks = 10000
    model = load_model(image_model, imageSize=imageSize, num_classes=num_tasks)

    #print("++++++++++++++++++++++++++")
    if resume:
        if os.path.isfile(resume):  # only support ResNet18 when loading resume
            #print("=> loading checkpoint '{}'".format(resume))
            if torch.cuda.is_available():
              checkpoint = torch.load(resume)
            else:
              checkpoint = torch.load(resume, map_location=torch.device('cpu'))
            ckp_keys = list(checkpoint['state_dict'])
            cur_keys = list(model.state_dict())
            model_sd = model.state_dict()
            if image_model == "ResNet18":
                ckp_keys = ckp_keys[:120]
                cur_keys = cur_keys[:120]

            for ckp_key, cur_key in zip(ckp_keys, cur_keys):
                model_sd[cur_key] = checkpoint['state_dict'][ckp_key]

            model.load_state_dict(model_sd)
            arch = checkpoint['arch']
            #print("resume model info: arch: {}".format(arch))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
    
    if torch.cuda.is_available():
      model = model.cuda()
    else:
      model = model

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transformer_test = [transforms.CenterCrop(imageSize), transforms.ToTensor()]

    names = load_filenames_and_labels_multitask2(txt_file=input_file)

    names = np.array(names)
    num_tasks = len(names)
    test_dataset = ImageDataset2(names, img_transformer=transforms.Compose(img_transformer_test),
                                normalize=normalize, ret_index=False, args=None)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=64,
                                                  shuffle=False,
                                                  num_workers=workers,
                                                  pin_memory=True)


    # Extract embeddings from the last layer of predictions
    embeddings = []
    with torch.no_grad():
        model.eval()
        for images in test_dataloader:
            images = images.to(device)
            outputs = model(images)
            embeddings.append(outputs.cpu().numpy())

    # Concatenate embeddings from all batches
    embeddings = np.concatenate(embeddings, axis=0)
    df = pd.DataFrame(embeddings)
    df = df.add_prefix('ImageMol_')
    filenames_without_extension = [get_filename_without_extension(path) for path in names]

    ids = filenames_without_extension
    # writeEmbeddingsIntoFile(".", f'ImageMol.csv', ids, embeddings)

    df.index = filenames_without_extension
    df.index.name = 'id'
    df.to_csv(output_file_name)




