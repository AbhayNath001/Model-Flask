from flask import Flask, render_template, request
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Lipinski, DataStructs
from difflib import SequenceMatcher

# Load dataset
df = pd.read_excel("Final Dataset.xlsx", usecols=[1])

app = Flask(__name__)

# Function to check Lipinski's Rule of Five with scoring
def check_lipinski_rule(input_smiles):
    mol = Chem.MolFromSmiles(input_smiles)
    mw = Descriptors.MolWt(mol)
    logp = Chem.Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)

    criteria = [
        mw <= 500,
        logp <= 5,
        hbd <= 5,
        hba <= 10
    ]
    score = criteria.count(True) / len(criteria)

    return {
        "MW": mw,
        "LogP": logp,
        "HBD": hbd,
        "HBA": hba,
        "Lipinski Score": score
    }

# Simulate ADMET evaluation based on molecular properties
def admet_evaluation_simulate(input_smiles):
    mol = Chem.MolFromSmiles(input_smiles)
    logp = Chem.Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)

    results = {
        "LogP": logp,
        "TPSA": tpsa,
        "MW": mw,
        "HBD": hbd,
        "HBA": hba,
        "Absorption": "Good" if logp <= 5 and tpsa <= 140 else "Poor",
        "Distribution": "Extensive" if mw <= 500 and hbd <= 5 else "Limited",
        "Metabolism": "Slow" if mw >= 300 else "Fast",
        "Excretion": "Normal" if tpsa <= 120 else "Delayed",
        "Toxicity": "Low" if hba <= 10 and logp <= 5 else "High",
        "Favorable": hba <= 10 and logp <= 5
    }

    return results

# Calculate Tanimoto Similarity
def calculate_similarity(input_smiles, dataset_smiles):
    mol1 = Chem.MolFromSmiles(input_smiles)
    mol2 = Chem.MolFromSmiles(dataset_smiles)
    if mol1 and mol2:
        fps1 = Chem.RDKFingerprint(mol1)
        fps2 = Chem.RDKFingerprint(mol2)
        return DataStructs.TanimotoSimilarity(fps1, fps2)
    return 0.0

# Calculate Morgan Fingerprint Similarity
def calculate_similarity_Morgan(input_smiles, dataset_smiles):
    mol1 = Chem.MolFromSmiles(input_smiles)
    mol2 = Chem.MolFromSmiles(dataset_smiles)
    if mol1 and mol2:
        fps1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
        fps2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
        return DataStructs.TanimotoSimilarity(fps1, fps2)
    return 0.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    input_smiles = request.form['smiles']
    tanimoto_threshold = 0.75
    found = False
    results = {}

    for index, row in df.iterrows():
        smiles = row['SMILES']
        tanimoto_similarity = calculate_similarity(input_smiles, smiles)
        tanimoto_similarity_Morgan = calculate_similarity_Morgan(input_smiles, smiles)
        string_similarity = SequenceMatcher(None, input_smiles, smiles).ratio()

        if tanimoto_similarity >= tanimoto_threshold:
            found = True
            results["Classification"] = "Cancer-related drug"
            results["Tanimoto Similarity"] = tanimoto_similarity
            results["Morgan Tanimoto Similarity"] = tanimoto_similarity_Morgan
            results["String Similarity"] = string_similarity

            lipinski_results = check_lipinski_rule(input_smiles)
            results["Lipinski"] = lipinski_results

            admet_results = admet_evaluation_simulate(input_smiles)
            results["ADMET"] = admet_results
            break

    if not found:
        results["Classification"] = "Not a cancer-related drug"
        results["Lipinski"] = check_lipinski_rule(input_smiles)
        results["ADMET"] = admet_evaluation_simulate(input_smiles)

    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
