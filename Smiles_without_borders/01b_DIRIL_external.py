# Initialize pandarallel
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

# Now you can import your module
from smiles_without_borders import process_dataset


df = pd.read_csv("../DIRIL/data/DIRIL_curated.csv")
df_processed = process_dataset(df, 'smiles', 'DIRIL_label', min_ph=7.35, max_ph=7.45)
df_processed.to_csv("../DIRIL/data/DIRIL_processed.csv")