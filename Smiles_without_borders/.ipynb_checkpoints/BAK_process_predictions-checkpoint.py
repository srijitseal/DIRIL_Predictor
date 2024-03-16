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



def main(opt):
	input_file = opt.input_file
	output_file = opt.output_file
	models = [1,2,3,4,5,6]
	beam = opt.beam_size

	pred_lines = {}
	predictions =set()

	for num in range(0,len(models)):
		predictions_file = 'model_'+str(models[num])+'_'+ output_file
		with open(predictions_file) as f_pred:  
			pred_lines[num] = [''.join(line.strip().split(' ')) for line in f_pred.readlines()]

	models_count = len(pred_lines.keys())

	molID2metabolites = {}
	index = 0

	smiles = pd.read_csv(input_file)["smiles"][0]
    
	if not check_smile(smiles):
		return
	smiles = canonicalise_smile(smiles)
	
	for j in range(index,index+beam):
		for num in range(0,models_count):
			predictions.add(pred_lines[num][j])        
	index = index + beam
	processed, invalid, invalid_count = process_predictions(predictions,smiles,0.25,0.25,False,True)

	molID2metabolites = processed
	drug = Chem.MolFromSmiles(smiles)
	preds = [pred_smiles for pred_smiles in processed]  
        
        
	df = pd.DataFrame({
	"Input": [smiles] * len(preds),  # Repeat the smiles string to match the length of preds
	"Metabolites": preds})


	df.to_csv(output_file, index=False)
  


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-input_file', type=str,help='Input File')
	parser.add_argument('-output_file', type=str, default='predicted_metabolites.csv',help='Processed Predictions File')
	parser.add_argument('-predictions_dir', type=str, default='predictions/',help='Predictions Directory')
	parser.add_argument('-beam_size', type=int, default=5,help='Beam Size')
	parser.add_argument('-visualise_molecules', type=bool, default=False,help='Visualise predicted metabolites')
	opt = parser.parse_args()
	main(opt)