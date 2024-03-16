import pandas as pd

from dimorphite_dl.dimorphite_dl import DimorphiteDL
from rdkit.Chem import AddHs
from rdkit.Chem import inchi
from rdkit import Chem
from rdkit.Chem import Draw
from IPython.display import display
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.MolStandardize import Standardizer
from rdkit.Chem import MolStandardize
from collections import Counter

import subprocess
import pandas as pd
from process_input_file import tokenizer
from gather_metabolites import gather_preds

from rdkit import Chem
from rdkit.Chem import inchi
from dimorphite_dl.dimorphite_dl import DimorphiteDL

from pandarallel import pandarallel
# Initialize pandarallel
pandarallel.initialize(progress_bar=True)

def smiles_to_inchikey14(smiles):
    """
    Converts a SMILES string to the first 14 characters of its corresponding InChIKey.

    Parameters:
    - smiles (str): A SMILES string representing a chemical compound.

    Returns:
    - str: The first 14 characters of the InChIKey corresponding to the input SMILES, or
           an empty string if conversion is not possible.
    """
    # Convert SMILES to a molecule object
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # Convert molecule to InChI
        inchi_val = inchi.MolToInchi(mol)
        if inchi_val:
            # Convert InChI to InChIKey
            inchi_key = inchi.InchiToInchiKey(inchi_val)
            # Return the first 14 characters of the InChIKey
            return inchi_key[:14]
    return ""

def safely_protonate(x, min_ph=7, max_ph=7):
    
    dimorphite = DimorphiteDL(min_ph=min_ph, max_ph=max_ph, pka_precision=0)
    
    result = dimorphite.protonate(x)
    # Check if result is a list and not empty
    if isinstance(result, list) and result:
        return result[0]  # Return first element if it's a list and has elements
    elif result:  # If result is not a list but not None or False (assuming dimorphite.protonate might return useful non-list values)
        return result
    return x  # Return the original value if result is None or an empty list


def process_dataset(df, smiles_column="smiles", binary_column="binary_label", min_ph=7, max_ph=7):
    
    # The result is a DataFrame where each cell contains another DataFrame of predictions
    df['Standardized_SMILES'] = df[smiles_column].parallel_apply(smiles_standardiser)
    df['Standardized_SMILES'] = df['Standardized_SMILES'].parallel_apply(check_carbon)
    # Filter out SMILES strings that couldn't be standardized
    filtered_data = df[df['Standardized_SMILES'] != "Cannot_do"]
    
    results_series = filtered_data.apply(lambda row: predict_metabolites_for_smiles(row['Standardized_SMILES'], row[smiles_column], row[binary_column], min_ph=min_ph, max_ph=max_ph), axis=1)

    # Combine the individual DataFrames into a single DataFrame
    # This involves concatenating the DataFrames stored in each cell of the results_series
    combined_results_df = pd.concat(results_series.values.flatten().tolist(), keys=results_series.index).reset_index(level=1, drop=True)
    
    return(combined_results_df)

def predict_metabolites_for_smiles(smiles_std, input_smiles, label, min_ph=7, max_ph=7, beam_size=5, visualize=False):
    
    try: 
        # Step 1: Call prepare_input_file.py to tokenize smiles
        smiles_processed = tokenizer(smiles_std)

        # Define the output path for the predicted metabolites
        processed_input_path = 'tmprw_xge6h_processed.csv'
        # Open the file in write mode ('w') and write the processed SMILES string to it
        with open(processed_input_path, 'w') as file:
            file.write(smiles_processed)

        predicted_metabolites_file = 'tmprw_xge6h_predicted_metabolites.csv'

        # Step 2: Translate the molecules into metabolites
        # Adjust your script or method to translate molecules based on the processed input
        # For the purpose of this example, we'll assume a script or function that can be called here.
        # This is where you'd integrate or call your translation mechanism.
        subprocess.run(['./translate_molecules', processed_input_path, str(beam_size), predicted_metabolites_file])

        #  3: Process predictions to get a CSV file of predicted metabolites
        # This might be redundant if your translation step already outputs a CSV; adjust as necessary.
        predicted_metabolites_df = gather_preds(smiles_std, predicted_metabolites_file, beam_size)

        # Creating a new row with the SMILES string for both columns
        original_smiles = pd.DataFrame({'Input_SMILES': [input_smiles], 'Input_Standardised_SMILES': [smiles_std], 'Output': [smiles_std], 'Label': [label]})

        # predicted_metabolites_df['Input_Standardised_SMILES'] = smiles_std
        predicted_metabolites_df['Label'] = label
        predicted_metabolites_df['Input_SMILES'] = input_smiles

        # Appending the original DataFrame to the new row DataFrame
        # The ignore_index=True option is used to reindex the new DataFrame
        final_df = pd.concat([original_smiles, predicted_metabolites_df], ignore_index=True)

        final_df["protonated_Output"] = final_df["Output"].apply(lambda x: safely_protonate(x, min_ph=min_ph, max_ph=max_ph))

        final_df["InchiKey"]= final_df["Output"].apply(smiles_to_inchikey14)
        final_df = final_df.drop_duplicates(subset=["InchiKey"], keep="first").reset_index(drop=True)

        return final_df
    
    except:
        
        return 

def smiles_standardiser(smiles, min_ph=7, max_ph=7):
    standardizer = Standardizer()
    smiles_original = smiles

    # Read SMILES and convert it to RDKit mol object
    mol = Chem.MolFromSmiles(smiles)
 
    try:
        smiles_clean_counter = Counter()
        mol_dict = {}
        is_finalize = False

        #This solved phosphate oxidation in most cases but introduces a problem for some compounds: eg. geldanamycin where the stable strcutre is returned
        inchi_standardised = Chem.MolToInchi(mol)
        mol = Chem.MolFromInchi(inchi_standardised)

        #display(Draw.MolToImage(mol))

        # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
        mol = rdMolStandardize.Cleanup(mol) 
        # if many fragments, get the "parent" (the actual mol we are interested in) 
        mol = rdMolStandardize.FragmentParent(mol)
        # try to neutralize molecule
        uncharger = rdMolStandardize.Uncharger() # annoying, but necessary as no convenience method exists

        mol = uncharger.uncharge(mol)# standardize molecules using MolVS and RDKit
        mol = standardizer.charge_parent(mol)
        mol = standardizer.isotope_parent(mol)
        mol = standardizer.stereo_parent(mol)

        #Normalize tautomers 
        normalizer = MolStandardize.tautomer.TautomerCanonicalizer()
        mol = normalizer.canonicalize(mol)

        #Final Rules
        mol = standardizer.standardize(mol)
        mol_standardized = mol

        # convert mol object back to SMILES
        smiles_standardized = Chem.MolToSmiles(mol_standardized)
                
        #print(smiles_standardized)
        return smiles_standardized 
    
    except:     
        
        return "Cannot_do"
    
def check_carbon(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                return smiles
    return "Cannot_do"