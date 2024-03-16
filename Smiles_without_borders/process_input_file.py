import argparse
from utils import *

##  it prepares the molecules for translation: it canonicalizes SMILES and then tokenizes them. 

def tokenizer(smiles):
    if not check_smile(smiles):
        print('invalid SMILES: ', smiles)
        return("Invalid")
    
    smiles = canonicalise_smile(smiles)
    smiles_tok = smi_tokenizer(smiles)
    
    return(smiles_tok)

