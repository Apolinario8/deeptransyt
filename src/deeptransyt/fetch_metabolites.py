# import requests
# from rdkit import Chem
# import re

def kegg_to_pubchem_id(kegg_id):
    """Fetch PubChem ID from KEGG using the KEGG ID."""
    url = f"http://rest.kegg.jp/get/{kegg_id}"
    response = requests.get(url)
    if response.status_code == 200:
        for line in response.text.splitlines():
            if "PubChem" in line:
                return line.split()[1]  
    return None

def pubchem_to_smiles(pubchem_id):
    """Fetch SMILES from PubChem using the PubChem ID."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{pubchem_id}/property/CanonicalSMILES/TXT"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text.strip()
    return None

def get_smiles_from_kegg_id(kegg_id):
    """Get SMILES representation for a given KEGG ID by cross-referencing with PubChem."""
    pubchem_id = kegg_to_pubchem_id(kegg_id)
    if pubchem_id:
        return pubchem_to_smiles(pubchem_id)
    return None

def get_metabolites_for_genome(genome_id):
    """Retrieve KEGG metabolite IDs associated with a given genome."""
    base_url = "http://rest.kegg.jp"
    pathways_url = f"{base_url}/list/pathway/{genome_id}"
    response = requests.get(pathways_url)
    
    if response.status_code != 200:
        return None
    
    pathways_data = response.text.splitlines()
    metabolites = set()
    
    for pathway in pathways_data:
        pathway_id = pathway.split("\t")[0].replace("path:", "")
        pathway_url = f"{base_url}/get/{pathway_id}"
        pathway_response = requests.get(pathway_url)
        
        if pathway_response.status_code != 200:
            continue
        
        pathway_lines = pathway_response.text.splitlines()
        for line in pathway_lines:
            if line.startswith("COMPOUND"):
                compound_id = line.split()[1]
                metabolites.add(compound_id)
    
    return metabolites

def convert_to_smiles(identifier):
    """Convert KEGG ID, InChI, or SMILES to a consistent SMILES format."""
    kegg_id_pattern = re.compile(r"^C\d+$")
    
    if kegg_id_pattern.match(identifier):  
        return get_smiles_from_kegg_id(identifier)
    elif identifier.startswith("InChI="):
        mol = Chem.inchi.MolFromInchi(identifier)
        return Chem.MolToSmiles(mol) if mol else None
    else:  
        mol = Chem.MolFromSmiles(identifier)
        return Chem.MolToSmiles(mol) if mol else None

def get_genome_metabolites_as_smiles(genome_id):
    """Retrieve all metabolites for a genome and convert them to SMILES format."""
    metabolites = get_metabolites_for_genome(genome_id)
    smiles_dict = {kegg_id: convert_to_smiles(kegg_id) for kegg_id in metabolites if convert_to_smiles(kegg_id)}
    return smiles_dict

