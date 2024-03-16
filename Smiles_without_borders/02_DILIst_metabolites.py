# Initialize pandarallel
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

# Now you can import your module
from smiles_without_borders import process_dataset


df = pd.read_csv("../DILIst/data/DILIst_curated.csv")
df_processed = process_dataset(df, 'smiles_r', 'TOXICITY', min_ph=7.35, max_ph=7.45)
df_processed.to_csv("../DILIst/data/DILIst_processed.csv")