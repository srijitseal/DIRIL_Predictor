import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from utils import *
import os

## it reads the output of the models, un-tokenises the predicted sequences and filters out unlikely metabolites
## -input_file: the csv file that has the input molecules (molecule ID and SMILES representations)
## -output_file: the filename where the processed predictions will be saved. It's a csv file. 
## predictions_directory: the directory where the output of the models from the tranaslate_molecules script is saved
## -beam_size: the beam_size. It can be in [5,10,15,20]
## -visualise_molecules (boolean): it visualises all predicted metabolites if True. They are stored within the predictions directory.

def gather_preds(smiles_std, predicted_metabolites_file, beam):

	models = [1,2,3,4,5,6]

	pred_lines = {}
	predictions =set()

	for num in range(0,len(models)):
		predictions_file = 'model_'+str(models[num])+'_'+ predicted_metabolites_file
		with open(predictions_file) as f_pred:  
			pred_lines[num] = [''.join(line.strip().split(' ')) for line in f_pred.readlines()]

	models_count = len(pred_lines.keys())

	molID2metabolites = {}
	index = 0

	for j in range(index,index+beam):
		for num in range(0,models_count):
			predictions.add(pred_lines[num][j])        
	index = index + beam
	processed, invalid, invalid_count = process_predictions(predictions,smiles_std,0.25,0.25,False,True)

	molID2metabolites = processed
	drug = Chem.MolFromSmiles(smiles_std)
	preds = [pred_smiles for pred_smiles in processed]  
        
        
	df = pd.DataFrame({
	"Input_Standardised_SMILES": [smiles_std] * len(preds),  # Repeat the smiles string to match the length of preds
	"Output": preds})


	return(df)
    
    