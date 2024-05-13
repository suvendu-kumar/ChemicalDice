import os
from mordred import Calculator, descriptors
import pandas as pd
from rdkit import Chem
from tqdm import tqdm


def descriptor_calculator(input_file,output_file):
  """
  Calculate descriptors for a given set of SMILES strings and save the results to a CSV file.

  Parameters
  ----------
  input_file : str
      The path to the input CSV file. The file should contain a column 'SMILES' with SMILES strings, a column 'id' with unique identifiers, and a column 'sdf_files' with the paths to the corresponding SDF files.
  output_file : str
      The path to the output CSV file where the calculated descriptors will be saved.

  Returns
  -------
  None

  Notes
  -----
  The function uses the Calculator class to calculate descriptors.
  The resulting descriptors are saved to a CSV file with the columns 'id', 'SMILES', and descriptor columns.
  If an error occurs during descriptor calculation for a molecule, an error message is printed and the calculation continues with the next molecule.
  """
  smiles_df = pd.read_csv(input_file)
  sdffile_name_list = smiles_df['sdf_files']
  id_list = smiles_df['id']
  smiles_list = smiles_df['SMILES']
  calc = Calculator(descriptors, ignore_3D=False)
  desc_columns=[str(d) for d in calc.descriptors]
  f = open(output_file,"w")
  f.write("id,SMILES,")
  header = ",".join(desc_columns)
  f.write(header)
  f.write("\n")
  for sdffile_name,id,smile in tqdm(zip(sdffile_name_list, id_list, smiles_list)):
    try:
      suppl = Chem.SDMolSupplier(sdffile_name)
      Des = calc(suppl[0])
      lst = []
      lst.append(id)
      lst.append(smile)
      for i in range(len(Des)):
        myVariable =Des[i]
        if type(myVariable) == int or type(myVariable) == float or str(type(myVariable)) == "<class 'numpy.float64'>":
          lst.append(Des[i])
        else:
          lst.append(None)
      lst = [str(x) for x in lst]
      row_str=",".join(lst)
      f.write(row_str)
      f.write("\n")
      #print("=",end="")
    except Exception as e:
      print(" Error in descriptor calculation",end="\t")
      print(id)
      print(e)
  f.close()

