import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski, DataStructs, Crippen
from difflib import SequenceMatcher

# Set base directory and load dataset using an absolute path
basedir = os.path.abspath(os.path.dirname(__file__))
df_path = os.path.join(basedir, "Final Dataset.xlsx")
df = pd.read_excel(df_path, usecols=[1])

app = Flask(__name__)

# Function to check Lipinski's Rule of Five with violation count
def check_lipinski_rule(input_smiles):
    mol = Chem.MolFromSmiles(input_smiles)
    if mol is None:
        return None  # Invalid SMILES
    mw = round(Descriptors.MolWt(mol), 2)
    logp = round(Chem.Crippen.MolLogP(mol), 2)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    criteria = [mw <= 500, logp <= 5, hbd <= 5, hba <= 10]
    violations = criteria.count(False)
    return {
        "MW": mw,
        "LogP": logp,
        "HBD": hbd,
        "HBA": hba,
        "Violations": violations
    }

# Function to calculate additional descriptors
def calculate_other_descriptors(input_smiles):
    mol = Chem.MolFromSmiles(input_smiles)
    if mol is None:
        return None  # Invalid SMILES
    tpsa = round(Descriptors.TPSA(mol), 2)
    nrb = Descriptors.NumRotatableBonds(mol)        # Number of Rotatable Bonds
    nar = Descriptors.NumAromaticRings(mol)           # Number of Aromatic Rings
    nha = mol.GetNumHeavyAtoms()                      # Number of Heavy Atoms
    num_ring_atoms = sum(len(ring) for ring in mol.GetRingInfo().BondRings())
    num_steric_atoms = nha - num_ring_atoms           # Number of Steric Atoms
    return {
        "TPSA": tpsa,
        "NRB": nrb,
        "NAR": nar,
        "NHA": nha,
        "Steric Atoms": num_steric_atoms
    }

# Calculate Tanimoto Similarity using RDKit's RDKFingerprint
def calculate_similarity(input_smiles, dataset_smiles):
    mol1 = Chem.MolFromSmiles(input_smiles)
    mol2 = Chem.MolFromSmiles(dataset_smiles)
    if mol1 and mol2:
        fps1 = Chem.RDKFingerprint(mol1)
        fps2 = Chem.RDKFingerprint(mol2)
        return round(DataStructs.TanimotoSimilarity(fps1, fps2), 2)
    return 0.0

# Calculate Morgan Fingerprint Similarity
def calculate_similarity_Morgan(input_smiles, dataset_smiles):
    mol1 = Chem.MolFromSmiles(input_smiles)
    mol2 = Chem.MolFromSmiles(dataset_smiles)
    if mol1 and mol2:
        fps1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
        fps2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
        return round(DataStructs.TanimotoSimilarity(fps1, fps2), 2)
    return 0.0

@app.route('/')
def index():
    return render_template('evaluation.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    input_smiles = request.form['smiles']
    results = {"Input SMILES": input_smiles}
    max_tanimoto = 0.0
    best_match = None

    # Find the best match (maximum Tanimoto similarity) from the dataset
    for index, row in df.iterrows():
        smiles = row['SMILES']
        current_tanimoto = calculate_similarity(input_smiles, smiles)
        if current_tanimoto > max_tanimoto:
            max_tanimoto = current_tanimoto
            best_match = smiles

    # If no best match found, use the input SMILES
    if best_match is None:
        best_match = input_smiles

    # Determine classification based on max_tanimoto thresholds
    if max_tanimoto >= 0.58:
        classification = "Cancer-related drug (High Chance)"
    elif 0.40 <= max_tanimoto < 0.58:
        classification = "Cancer-related drug (Low Chance)"
    else:
        classification = "Not a Cancer-related drug"

    results["Classification"] = classification
    results["Tanimoto Similarity"] = max_tanimoto
    results["Morgan Tanimoto Similarity"] = calculate_similarity_Morgan(input_smiles, best_match)
    results["String Similarity"] = round(SequenceMatcher(None, input_smiles, best_match).ratio(), 2)
    results["Lipinski"] = check_lipinski_rule(input_smiles)
    results["Other Descriptors"] = calculate_other_descriptors(input_smiles)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
