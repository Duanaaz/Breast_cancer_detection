import pandas as pd
import numpy as np

# Metabolite names and their fold changes (from Table 1)
metabolites = [
    "L-glutamate", "L-alanine", "Glycerol_3_phosphate", "Succinate",
    "20-hydroxy-PGE2", "Fosfomycin", "3-methyluridine", "N-acetyl-L-alanine",
    "Choline", "Trigonelline", "Ile-Asn", "Arachidonic_acid",
    "S-methyl-5-thioadenosine", "Creatinine", "L-histidinol",
    "Guanidine_acetic_acid", "Cytosine", "Inosine", "Adenine", "Hypoxanthine"
]

# Fold changes for FMC-positive samples (from Table 1)
fold_changes = {
    "L-glutamate": 2.30, "L-alanine": 1.90, "Glycerol_3_phosphate": 1.59,
    "Succinate": 1.45, "20-hydroxy-PGE2": 2.08, "Fosfomycin": 2.04,
    "3-methyluridine": 1.75, "N-acetyl-L-alanine": 1.70, "Choline": 1.78,
    "Trigonelline": 1.61, "Ile-Asn": 1.72, "Arachidonic_acid": 1.56,
    "S-methyl-5-thioadenosine": 1.64, "Creatinine": 1.57, "L-histidinol": 1.60,
    "Guanidine_acetic_acid": 1.65, "Cytosine": 1.69, "Inosine": -1.52,
    "Adenine": 1.52, "Hypoxanthine": 1.59
}

# Generate synthetic data
np.random.seed(42)  # For reproducibility
n_samples = 300  # Total samples (150 FMC-positive, 150 healthy)
data = []

for _ in range(n_samples // 2):
    # FMC-positive samples (label = 1)
    row = []
    for metab in metabolites:
        if metab == "Inosine":
            base_value = np.random.uniform(0.6, 0.9)  # Decreased in FMC
        else:
            base_value = np.random.uniform(1.3, 2.4)  # Increased in FMC
        row.append(base_value)
    row.append(1)
    data.append(row)

    # Healthy samples (label = 0)
    row = []
    for metab in metabolites:
        if metab == "Inosine":
            base_value = np.random.uniform(1.1, 1.5)  # Normal/higher in healthy
        else:
            base_value = np.random.uniform(0.5, 1.2)  # Normal/lower in healthy
        row.append(base_value)
    row.append(0)
    data.append(row)

# Create DataFrame
df = pd.DataFrame(data, columns=metabolites + ["label"])

# Save to CSV
df.to_csv("fmc_dataset_expanded.csv", index=False)
print("Dataset generated as 'fmc_dataset_expanded.csv'")