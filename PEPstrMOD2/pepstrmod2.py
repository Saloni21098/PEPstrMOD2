# Structure Prediction of Chemically Modified Peptides Map Sequence (single or multiple)
# By Prof. Raghava's Group, Computational Biology Department, IIITD
# 14-07-2025


#Import necessary libraries
import argparse
import re
import subprocess
import shutil
import numpy as np
import matplotlib.pyplot as plt
import requests
import time
import zipfile
import os
import platform
from colabfold.batch import run
import traceback
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import PPBuilder
from available_ptms import AVAILABLE_PTMS, PTM_LOOKUP, resolve_ptm
from available_ncaas import AVAILABLE_NCAAS, MODCODE_LOOKUP, TAG_TO_BASE
from available_terminal_mods import USER_FRIENDLY_TERMINAL_MODS, resolve_terminal_mod
from ffptm_atom_templates import parse_ffptm
from ffncaa_atom_templates import parse_ffncaa


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FFNCAA_PATH = os.path.join(SCRIPT_DIR, "fixed_ffncaa")
TEMPLATE_PATH = os.path.join(SCRIPT_DIR, "templates")

ALLOWED_ATOMS_PER_RES, ATOM_RENAME_MAP = parse_ffncaa(FFNCAA_PATH, TEMPLATE_PATH)

D_RESI_MAP = {
    "ALA": "DAL", "ARG": "DAR", "ASN": "DSG", "ASP": "DAS", "CYS": "DCY",
    "GLN": "DGN", "GLU": "DGL", "HIS": "DHI", "ILE": "DIL",
    "LEU": "DLE", "LYS": "DLY", "MET": "MED", "PHE": "DPNs", "PRO": "DPR",
    "SER": "DSN", "THR": "DTH", "TRP": "DTR", "TYR": "DTY", "VAL": "DVA"
}

ONE_TO_THREE = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
}


AA_MAP = {
    "A": "ALA", "ALA": "ALA", "ALANINE": "ALA",
    "R": "ARG", "ARG": "ARG", "ARGININE": "ARG",
    "N": "ASN", "ASN": "ASN", "ASPARAGINE": "ASN",
    "D": "ASP", "ASP": "ASP", "ASPARTIC": "ASP",
    "C": "CYS", "CYS": "CYS", "CYSTEINE": "CYS",
    "E": "GLU", "GLU": "GLU", "GLUTAMIC": "GLU",
    "Q": "GLN", "GLN": "GLN", "GLUTAMINE": "GLN",
    "G": "GLY", "GLY": "GLY", "GLYCINE": "GLY",
    "H": "HIS", "HIS": "HIS", "HISTIDINE": "HIS",
    "I": "ILE", "ILE": "ILE", "ISOLEUCINE": "ILE",
    "L": "LEU", "LEU": "LEU", "LEUCINE": "LEU",
    "K": "LYS", "LYS": "LYS", "LYSINE": "LYS",
    "M": "MET", "MET": "MET", "METHIONINE": "MET",
    "F": "PHE", "PHE": "PHE", "PHENYLALANINE": "PHE",
    "P": "PRO", "PRO": "PRO", "PROLINE": "PRO",
    "S": "SER", "SER": "SER", "SERINE": "SER",
    "T": "THR", "THR": "THR", "THREONINE": "THR",
    "W": "TRP", "TRP": "TRP", "TRYPTOPHAN": "TRP",
    "Y": "TYR", "TYR": "TYR", "TYROSINE": "TYR",
    "V": "VAL", "VAL": "VAL", "VALINE": "VAL"
}


def detect_mpi_engine(max_cores_cap=8):
    total_cores = os.cpu_count() or 1   # fallback to 1 if None

    # Check if MPI version exists
    mpi_exists = shutil.which("pmemd.MPI") is not None

    if total_cores == 1 or not mpi_exists:
        print("Using serial pmemd (1 core)")
        return "pmemd"

    # Limit core usage (important for shared systems)
    use_cores = min(total_cores, max_cores_cap)

    print(f"Using MPI pmemd with {use_cores} cores")
    return f"mpirun -np {use_cores} pmemd.MPI"


def read_named_sequences(input_value, default_prefix):
    if os.path.isfile(input_value):
        entries, name, seq = [], None, ""
        with open(input_value) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith(">"):
                    if seq:
                        entries.append((name, seq))
                        seq = ""
                    name = line[1:].split()[0]
                else:
                    seq += line
        if seq:
            entries.append((name, seq))
        if not entries:
            raise ValueError("No sequences found in input file")
        for i, (n, s) in enumerate(entries):
            if not n:
                entries[i] = (f"{default_prefix}_{i+1}", s)
        return entries
    seq = input_value.strip()
    if not seq:
        raise ValueError("Empty sequence input")
    return [(default_prefix, seq)]


def validate_sequence_letters(seq):
    """
    Reject ambiguous or invalid natural amino acid codes
    before PTM/NNR/D-residue parsing.
    Allowed: A R N D C Q E G H I L K M F P S T W Y V
    Rejected: B J O U X Z
    """
    invalid = set("BJOUXZ")

    for i, aa in enumerate(seq, start=1):
        if aa.isalpha() and aa.upper() in invalid:
            raise ValueError(
                f"Invalid amino acid '{aa}' at position {i}. "
                f"'{aa.upper()}' is ambiguous or unsupported. "
                "Please replace it with a specific residue."
            )
            
            
def parse_sequence(raw_sequence):
    """
    Parses a raw peptide sequence with optional modifications and terminal caps.

    Returns:
        clean_seq: str of residues (1-letter)
        modifications: list of dicts (position, modification info)
        n_cap: N-terminal modification (or None)
        c_cap: C-terminal modification (or None)
        nc_cyclization: bool
        disulfide_pairs: list of (int, int)
    """

    import re

    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------
    clean_seq = ""
    modifications = []
    caps = {"N": None, "C": None}
    nc_cyclization = False
    disulfide_pairs = []

    raw_sequence = raw_sequence.replace("\n", "").replace(" ", "")

    # ------------------------------------------------------------
    # 1. Extract terminal caps
    # ------------------------------------------------------------
    term_pattern = r"\{(nt|ct):(.*?)\}"
    found_caps = re.findall(term_pattern, raw_sequence)

    seq_only = re.sub(term_pattern, "", raw_sequence)
    seq_only = re.sub(r"\{cyc:[^}]*\}", "", seq_only)

    letters = [c for c in seq_only if c.isalpha() and c.isupper()]
    if not letters:
        raise ValueError("No amino acids remain after removing caps/cyclization.")

    first_residue = letters[0]
    last_residue = letters[-1]

    raw_sequence = re.sub(term_pattern, "", raw_sequence)

    # ------------------------------------------------------------
    # 2. Process cyclization
    # ------------------------------------------------------------
    for annotation in re.findall(r"\{cyc:([^}]*)\}", raw_sequence):
        ann = annotation.strip().upper()
        if ann == "N-C":
            nc_cyclization = True
        else:
            for p in ann.split(","):
                a, b = map(int, p.strip().split("-"))
                disulfide_pairs.append((a, b))

    raw_sequence = re.sub(r"\{cyc:[^}]*\}", "", raw_sequence)

    # ------------------------------------------------------------
    # 3. Resolve terminal caps
    # ------------------------------------------------------------
    for term_tag, term_val in found_caps:
        terminal_type = term_tag.lower()
        residue_name = first_residue if terminal_type == "nt" else last_residue
        caps["N" if terminal_type == "nt" else "C"] = resolve_terminal_mod(
            term_val.strip().upper(), residue_name, terminal_type
        )

    # ------------------------------------------------------------
    # 4. Resolve NNR placeholders → parent residues
    # ------------------------------------------------------------
    processed = ""
    temp_nnr_mods = []

    i = 0
    while i < len(raw_sequence):
        if raw_sequence.startswith("{nnr:", i):
            end_idx = raw_sequence.find("}", i)
            if end_idx == -1:
                raise ValueError("Unclosed {nnr:TAG} block.")

            tag = raw_sequence[i+5:end_idx].strip().upper()
            base_res = TAG_TO_BASE.get(tag)

            if not base_res:
                raise ValueError(f"Unknown NNR tag '{tag}'.")

            if base_res not in AVAILABLE_NCAAS or tag not in AVAILABLE_NCAAS[base_res]:
                raise ValueError(f"NNR '{tag}' not defined for residue '{base_res}'.")

            position = len(processed) + 1
            processed += base_res

            temp_nnr_mods.append({
                "position": position,
                "residue": base_res,
                "modification": AVAILABLE_NCAAS[base_res][tag]["mod_code"],
                "mod_type": "nnr"
            })

            i = end_idx + 1
        else:
            processed += raw_sequence[i]
            i += 1

    raw_sequence = processed

    # ------------------------------------------------------------
    # 5. Main parsing loop (PTM / D-residues)
    # ------------------------------------------------------------
    idx = 0
    i = 0

    while i < len(raw_sequence):
        ch = raw_sequence[i]

        if not (ch.isalpha() and ch.isupper()):
            raise ValueError(f"Unexpected character '{ch}' at index {i}")

        residue = ch
        mod = None

        if i + 1 < len(raw_sequence) and raw_sequence[i + 1] == "{":
            end_idx = raw_sequence.find("}", i + 2)
            if end_idx == -1:
                raise ValueError(f"Unclosed modification after residue '{residue}'")
            mod = raw_sequence[i + 2:end_idx]
            i = end_idx + 1
        else:
            i += 1

        idx += 1
        clean_seq += residue

        if not mod:
            continue

        mod_lower = mod.lower()

        # D-amino acid
        if mod_lower == "d":
            if residue != "G":
                d_res = D_RESI_MAP.get(ONE_TO_THREE[residue])
                if not d_res:
                    raise ValueError(f"No D-residue mapping for {residue}")
                modifications.append({
                    "position": idx,
                    "residue": residue,
                    "modification": d_res,
                    "mod_type": "d"
                })

        # PTM
        elif mod_lower.startswith("ptm"):
            tag = mod.split(":", 1)[-1].strip()
            res_key = residue if residue in AVAILABLE_PTMS else ONE_TO_THREE[residue]
            resolved_code = resolve_ptm(res_key, tag)
            modifications.append({
                "position": idx,
                "residue": residue,
                "modification": resolved_code,
                "mod_type": "ptm"
            })

        else:
            raise ValueError(f"Unknown modification '{mod}' at position {idx}")

    # ------------------------------------------------------------
    # 6. Merge NNR modifications
    # ------------------------------------------------------------
    modifications.extend(temp_nnr_mods)

    # ------------------------------------------------------------
    # 7. Sanity check
    # ------------------------------------------------------------
    for m in modifications:
        if m["position"] < 1 or m["position"] > len(clean_seq):
            raise RuntimeError(
                f"Modification at position {m['position']} "
                f"exceeds sequence length {len(clean_seq)}"
            )

    return clean_seq, modifications, caps["N"], caps["C"], nc_cyclization, disulfide_pairs
    

def apply_terminal_modifications(sequence, caps):
    """
    Applies terminal modifications defined in caps dict.
    sequence: list of 3-letter residues (e.g. ['GLY','ALA',...])
    caps: dict with keys "N" and "C" containing mod codes (e.g. '3XH') or None

    NOTE: This function only updates the sequence list (in-memory). It does NOT
    add or remove residues. Additive caps (ACE, NME, etc.) are not inserted.
    """
    if not sequence or not isinstance(sequence, list):
        raise ValueError("apply_terminal_modifications expects sequence as a list of 3-letter codes.")

    seq = sequence.copy()

    # --- N-terminal modification ---
    if caps.get("N"):
        n_mod = caps["N"].upper()
        first_res = seq[0].upper()
        found = False

        # Search in user-friendly NT mapping to confirm validity
        nt_mods = USER_FRIENDLY_TERMINAL_MODS.get("nt", {})
        for alias, mapping in nt_mods.items():
            if first_res in mapping.values() or first_res in mapping.keys():
                if mapping.get(first_res) == n_mod:
                    found = True
                    break

        if not found:
            # We don’t have a strict "allowed" list anymore, so only print a warning
            print(f"⚠ Warning: N-terminal mod '{n_mod}' not found in Excel mapping for '{first_res}', applying anyway.")

        seq[0] = n_mod
        print(f"Applied N-terminal modification '{n_mod}' to {first_res}")

    # --- C-terminal modification ---
    if caps.get("C"):
        c_mod = caps["C"].upper()
        last_res = seq[-1].upper()
        found = False

        # Search in user-friendly CT mapping to confirm validity
        ct_mods = USER_FRIENDLY_TERMINAL_MODS.get("ct", {})
        for alias, mapping in ct_mods.items():
            if last_res in mapping.values() or last_res in mapping.keys():
                if mapping.get(last_res) == c_mod:
                    found = True
                    break

        if not found:
            print(f"Warning: C-terminal mod '{c_mod}' not found in Excel mapping for '{last_res}', applying anyway.")

        seq[-1] = c_mod
        print(f"Applied C-terminal modification '{c_mod}' to {last_res}")

    return seq


def setup_colabfold_params():
    """
    Copies model parameter files from the provided colabfold_params folder
    (one level above this script) into ~/.cache/colabfold/params/.
    """

    # Determine this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to colabfold_params (one directory above script_dir)
    project_root = os.path.dirname(script_dir)
    source_dir = os.path.join(project_root, "colabfold_params")

    # Path where ColabFold wants params
    target_dir = os.path.join(os.path.expanduser("~"), ".cache", "colabfold", "params")
    os.makedirs(target_dir, exist_ok=True)

    if not os.path.exists(source_dir):
        raise FileNotFoundError(
            f"[ERROR] Could not find colabfold_params folder at: {source_dir}"
        )

    #print(f"[PEPstrMOD2] Preparing ColabFold parameters...")
    copied = False

    for fname in os.listdir(source_dir):
        src = os.path.join(source_dir, fname)
        dst = os.path.join(target_dir, fname)

        if not os.path.exists(dst):
            shutil.copy(src, dst)
            #print(f"[PEPstrMOD2] Installed model file: {fname}")
            copied = True

    if not copied:
        print("[PEPstrMOD2] All model files already installed.")

    print(f"[PEPstrMOD2] Parameters available in: {target_dir}")
    
    
def predict_structure(sequence, out_dir, jobname="test_pred",
                      method="esmfold",  # or "alphafold2"
                      retries=3, delay=5):
    """
    Predicts protein structure using either ESMFold or alphafold2.
    
    If ESMFold fails (API or network error), it automatically falls back to alphafold2.
    
    Args:
        sequence (str): Amino acid sequence (natural letters).
        out_dir (str): Output directory to save the result.
        jobname (str): Base name for output files.
        method (str): "esmfold" or "alphafold2".
        retries (int): Number of retries for ESMFold API.
        delay (int): Delay (seconds) between retries.
    
    Returns:
        str: Path to predicted PDB file.
    """
    os.makedirs(out_dir, exist_ok=True)
    pdb_path = os.path.join(out_dir, f"{jobname}.pdb")

    # Helper: run ESMFold
    def run_esmfold():
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}

        for attempt in range(1, retries + 1):
            print(f"\nSubmitting to ESMFold (attempt {attempt})...")

            try:
                response = requests.post(
                    'https://api.esmatlas.com/foldSequence/v1/pdb/',
                    headers=headers,
                    data=sequence
                )

                if response.status_code == 200:
                    with open(pdb_path, 'w') as f:
                        f.write(response.text)
                    append_ter_end_to_file(pdb_path)
                    print(f"\n[✓] ESMFold structure saved to {pdb_path}")
                    return pdb_path
                else:
                    print(f"[Attempt {attempt}] API error: {response.status_code} — {response.text.strip()}")

            except requests.RequestException as e:
                print(f"[Attempt {attempt}] Request failed: {e}")

            if attempt < retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)

        raise RuntimeError("ESMFold API failed after multiple attempts")

    # Helper: run alphafold2
    def run_alphafold2():
        setup_colabfold_params()
        queries = [(jobname, sequence, None)]

        try:
            # Attempt full MSA run
            print("\nRunning alphafold2 with MSA mode...")
            run(
                queries=queries,
                result_dir=out_dir,
                msa_mode="mmseqs2_uniref_env",
                num_models=5,
                is_complex=False,
                user_agent="PEPstrMOD/2.0 salonij@iiitd.ac.in"
            )

        except Exception as e:
            print("\n[WARNING] MSA-based alphafold2 run failed!")
            print("Error:", e)
            traceback.print_exc()

            print("\nRetrying in single-sequence mode...")
            run(
                queries=queries,
                result_dir=out_dir,
                msa_mode="single_sequence",
                model_type="alphafold2_ptm",
                num_models=5,
                is_complex=False,
                user_agent="PEPstrMOD/2.0 salonij@iiitd.ac.in"
            )

        # --- Normalize alphafold2 output ---
        import glob
        import shutil

        pattern = os.path.join(out_dir, f"{jobname}_unrelaxed_rank_001*.pdb")
        found = glob.glob(pattern)

        if found:
            top_ranked = found[0]
            shutil.copy(top_ranked, pdb_path)
            print(f"[✓] Top alphafold2 model copied → {pdb_path}")
        else:
            candidates = [f for f in os.listdir(out_dir) if f.endswith(".pdb")]
            if candidates:
                fallback = os.path.join(out_dir, candidates[0])
                shutil.copy(fallback, pdb_path)
                print(f"[✓] Fallback alphafold2 model used → {pdb_path}")
            else:
                raise RuntimeError("alphafold2 did not produce a PDB file.")

        return pdb_path

    # --- Main control flow ---
    if method.lower() == "esmfold":
        try:
            return run_esmfold()
        except Exception as e:
            print("\n[WARNING] ESMFold prediction failed!")
            print("Error:", e)
            print("\n→ Falling back to alphafold2...")
            return run_alphafold2()

    elif method.lower() == "alphafold2":
        return run_alphafold2()

    else:
        raise ValueError("Invalid method. Choose either 'esmfold' or 'alphafold2'.")


def append_ter_end_to_file(pdb_path):
    """Appends TER and END lines to a PDB file based on the last ATOM entry."""
    with open(pdb_path, 'r') as f:
        lines = f.readlines()
    atom_lines = [l for l in lines if l.startswith("ATOM")]
    if not atom_lines:
        raise ValueError("No ATOM records found in PDB.")
    last = atom_lines[-1]
    serial = int(last[6:11])
    resname = last[17:20]
    chain = last[21]
    resnum = int(last[22:26])
    ter_line = f"\nTER   {serial + 1:5d}      {resname} {chain}{resnum:4d}\nEND\n"
    with open(pdb_path, 'a') as f:
        f.write(ter_line)


def clean_pdb(pdb_file, cleaned_filename, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, cleaned_filename)

    with open(pdb_file) as f:
        lines = f.readlines()

    cleaned = [
        l for l in lines
        if not l.lstrip().startswith(("CONECT", "MASTER", "ENDMDL"))
        and not l[12:16].strip().startswith("H")
    ]

    with open(output_path, 'w') as f:
        f.writelines(cleaned)

    print(f"PDB cleaned: {output_path}")
    return output_path  # return for convenience


def extract_sequence_oneletter(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", pdb_file)
    ppb = PPBuilder()
    sequence = ""
    for pp in ppb.build_peptides(structure):
        sequence += pp.get_sequence()
    return str(sequence)


def extract_sequence_threeletter(pdb_file):
    """
    Returns a list of 3-letter residue names in order from ATOM records.
    Skips water and heteroatoms.
    """
    seen_residues = set()
    seq = []

    with open(pdb_file) as f:
        for line in f:
            if line.startswith("ATOM"):
                resnum = int(line[22:26])
                resname = line[17:20].strip()
                chain = line[21]
                key = (chain, resnum)
                if key not in seen_residues:
                    seen_residues.add(key)
                    seq.append(resname)
    return seq


def parse_pdb_residues(pdb_file):
    """
    Extracts all residues and CYS SG atoms from a PDB.
    Returns:
        - residues: List of (resname, resid)
        - sg_atoms: List of (resid, coordinates as np.array)
    """
    residues, sg_atoms = [], []
    seen = set()

    with open(pdb_file) as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                resname = line[17:20].strip()
                resid = int(line[22:26])

                if resid not in seen:
                    residues.append((resname, resid))
                    seen.add(resid)

                if resname == "CYS" and atom_name == "SG":
                    x, y, z = map(float, (line[30:38], line[38:46], line[46:54]))
                    sg_atoms.append((resid, np.array([x, y, z])))

    return residues, sg_atoms


def build_sequence_string(residues):
    """Builds a space-separated string of residue names from (resname, resid) list."""
    return " ".join([res[0] for res in residues])


def get_atom_distance(pdb_file, res1, atom1, res2, atom2, label=None):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("mol", pdb_file)

    res_map = {}
    for model in structure:
        for chain in model:
            for res in chain:
                res_map[res.get_id()[1]] = res

    r1 = res_map.get(res1)
    r2 = res_map.get(res2)

    if not r1 or not r2:
        if label:
            print(f"{label}: Residue {res1} or {res2} not found.")
        return None

    if atom1 not in r1 or atom2 not in r2:
        if label:
            print(f"{label}: Atom '{atom1}' or '{atom2}' missing in residues {res1}, {res2}")
        return None

    dist = np.linalg.norm(r1[atom1].coord - r2[atom2].coord)
    if label:
        print(f"{label}: Distance between {r1.resname}{res1}-{atom1} and {r2.resname}{res2}-{atom2} = {dist:.2f} Å")
    return dist


def get_nc_distance(pdb_file, label=None):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("mol", pdb_file)
    
    residues = [res for res in structure.get_residues()
                if res.id[0] == " "]

    if len(residues) == 0:
        if label: print(f"{label}: no residues found")
        return None

    first = residues[0]
    last = residues[-1]

    # Find backbone N and C atoms directly
    N = first.get_atoms().__next__().get_parent()["N"] if "N" in first else None
    C = last["C"] if "C" in last else None
    
    if N is None or C is None:
        if label:
            print(f"{label}: Could not find N or C atom: N={N}, C={C}")
        return None

    # Compute distance
    dist = np.linalg.norm(N.coord - C.coord)
    return dist


def print_sg_distances(pdb_file, res_pairs, label=None, out_path=None):
    header = f"\n---Disulfide bond distance check{' - ' + label if label else ''}---"
    print(header)
    if out_path:
        with open(out_path, "a") as f:
            f.write(header + "\n")

    for res1, res2 in res_pairs:
        msg = f"SG distance CYS {res1}-{res2}"
        dist = get_atom_distance(pdb_file, res1, "SG", res2, "SG", msg)

        # Write to log if enabled and distance is valid
        if out_path and dist is not None:
            with open(out_path, "a") as f:
                f.write(f"{msg}: {dist:.2f} Å\n")


def detect_disulfides(sg_atoms):
    pairs = []
    for i, (res1, coord1) in enumerate(sg_atoms):
        for res2, coord2 in sg_atoms[i+1:]:
            if np.linalg.norm(coord1 - coord2) < 2.5:
                pairs.append((res1, res2))
    return pairs


def rename_cys_to_cyx_and_add_conect(pdb_file, disulfide_pairs, output_file):
    cyx_res = {r for pair in disulfide_pairs for r in pair}
    new_lines = []
    with open(pdb_file) as f:
        for line in f:
            if line.startswith("ATOM"):
                resid = int(line[22:26])
                atom_name = line[12:16].strip()
                if resid in cyx_res:
                    line = line[:17] + "CYX" + line[20:]
            new_lines.append(line)
    # Do NOT add CONECT lines — let tleap handle bonding
    with open(output_file, 'w') as f:
        f.writelines(new_lines)
    print(f"Disulfide processed (no CONECT): {output_file}")


ADDITIVE_CAPS = {"ACE", "NHE"}  # any classical additive caps you consider forbidden


def modify_pdb(pdb_file, replacements, output_file, nterm_mod=None, cterm_mod=None):
    """
    Modify PDB residues including internal replacements and terminal caps.

    - replacements: dict mapping residue number (int) -> 3-letter code (string)
      for internal replacements (residue numbering must match PDB).
    - nterm_mod / cterm_mod: 3-letter or user keyword (e.g. "Coxy", "acet")
      representing N- or C-terminal modifications.
    """

    from available_terminal_mods import resolve_terminal_mod

    # --- Validate inputs ---
    if nterm_mod:
        nterm_mod = str(nterm_mod).strip()
    if cterm_mod:
        cterm_mod = str(cterm_mod).strip()

    # --- Read the input PDB ---
    with open(pdb_file, "r") as f:
        lines = f.readlines()

    # --- Identify residue order (chain, residue_number) ---
    res_ids = []
    for ln in lines:
        if ln.startswith(("ATOM", "HETATM")):
            chain = ln[21]
            try:
                res_no = int(ln[22:26].strip())
            except ValueError:
                continue
            if (chain, res_no) not in res_ids:
                res_ids.append((chain, res_no))

    if not res_ids:
        raise ValueError("No residues found in PDB (missing ATOM/HETATM records).")

    first_res = res_ids[0]
    last_res = res_ids[-1]

    # --- Detect the first and last residue names from the PDB ---
    pdb_first_name = None
    pdb_last_name = None
    for ln in lines:
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        chain = ln[21]
        try:
            res_no = int(ln[22:26].strip())
        except ValueError:
            continue
        if (chain, res_no) == first_res and not pdb_first_name:
            pdb_first_name = ln[17:20].strip().upper()
        if (chain, res_no) == last_res and not pdb_last_name:
            pdb_last_name = ln[17:20].strip().upper()
        if pdb_first_name and pdb_last_name:
            break

    # --- Resolve terminal mods (using new resolver logic) ---
    nterm_final = None
    cterm_final = None

    if nterm_mod and nterm_mod.lower() != "none":
        try:
            nterm_final = resolve_terminal_mod(nterm_mod, pdb_first_name, "nt")
            print(f"N-terminal '{nterm_mod}' mapped to {nterm_final} for {pdb_first_name}")
        except Exception as e:
            raise ValueError(
                f"Invalid N-terminal mod '{nterm_mod}' for {pdb_first_name}: {e}"
            )

    if cterm_mod and cterm_mod.lower() != "none":
        try:
            cterm_final = resolve_terminal_mod(cterm_mod, pdb_last_name, "ct")
            print(f"C-terminal '{cterm_mod}' mapped to {cterm_final} for {pdb_last_name}")
        except Exception as e:
            raise ValueError(
                f"Invalid C-terminal mod '{cterm_mod}' for {pdb_last_name}: {e}"
            )

    # --- Apply replacements and terminal mods ---
    modified_lines = []
    for ln in lines:
        if ln.startswith(("ATOM", "HETATM")):
            chain = ln[21]
            try:
                res_no = int(ln[22:26].strip())
            except ValueError:
                modified_lines.append(ln)
                continue

            key = (chain, res_no)

            # Replace N-terminal residue
            if key == first_res and nterm_final:
                new_resname = nterm_final[:3].upper().ljust(3)
                ln = ln[:17] + new_resname + ln[20:]

            # Replace C-terminal residue
            elif key == last_res and cterm_final:
                new_resname = cterm_final[:3].upper().ljust(3)
                ln = ln[:17] + new_resname + ln[20:]

            # Replace internal modified residues
            elif res_no in (replacements or {}):
                new_name = str(replacements[res_no])[:3].upper().ljust(3)
                ln = ln[:17] + new_name + ln[20:]

        modified_lines.append(ln)

    # --- Write the modified PDB ---
    with open(output_file, "w") as fw:
        fw.writelines(modified_lines)

    print(f"Modified PDB saved to: {output_file}")


def extract_sequence_from_pdb(pdb_path):
    """
    Extracts the ordered list of residue names from a PDB file.
    Skips water, ions, or heteroatoms.

    Args:
        pdb_path (str): Path to the PDB file

    Returns:
        list[str]: Sequence of 3-letter residue names
    """
    residues = []
    seen = set()

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                res_name = line[17:20].strip()
                res_id = (line[21], int(line[22:26]))  # (chain, residue number)
                if res_id not in seen:
                    seen.add(res_id)
                    residues.append(res_name)

    return residues


def align_pdb_atoms_to_prepi(pdb_file, prepi_file, out_pdb):
    """
    Align a PDB residue’s atom names to match a PREPIN template.
    - Prefers atoms tagged with 'M' (main chain) for backbone mapping.
    - Renames N, CA, C, O to PREPIN equivalents.
    - Removes atoms not present in PREPIN to prevent tleap errors.
    """

    if not os.path.isfile(prepi_file):
        print(f"[WARN] PREPIN not found: {prepi_file}")
        return pdb_file

    atoms = []  # (idx, name, type, region_tag, [connects])
    resname = None
    in_atom_block = False

    # ---- Parse PREPIN ----
    with open(prepi_file) as f:
        for line in f:
            # Detect residue name line
            m = re.match(r"^\s*([A-Z0-9]{2,4})\s+INT", line)
            if m:
                resname = m.group(1)
                continue

            # Start of atom block
            if "CORRECT" in line and "OMIT" in line:
                in_atom_block = True
                continue

            # Stop parsing when loop/improper/done
            if in_atom_block and line.strip().startswith(("LOOP", "IMPROPER", "DONE")):
                in_atom_block = False

            if in_atom_block:
                parts = line.split()
                if len(parts) >= 7 and parts[1] != "DUMM":
                    try:
                        idx = int(parts[0])
                    except ValueError:
                        continue
                    name = parts[1].strip()
                    atype = parts[2].strip()
                    region = parts[3].strip() if len(parts) > 3 else ""
                    # Connectivity indices (usually 4–6)
                    connects = []
                    try:
                        connects = [int(parts[4]), int(parts[5]), int(parts[6])]
                    except Exception:
                        connects = []
                    atoms.append((idx, name, atype, region, connects))

    if not atoms:
        print(f"[WARN] No atoms parsed from PREPIN: {prepi_file}")
        return pdb_file

    # All atom names defined in PREPIN
    prepi_atom_names = {a[1] for a in atoms}

    # --- Helper: choose atom by type with preference for region 'M' ---
    def choose_atom(criteria):
        candidates = [a for a in atoms if criteria(a)]
        if not candidates:
            return None
        mains = [a for a in candidates if a[3] == "M"]
        return mains[0] if mains else candidates[0]

    bb_map = {}

    # N atom
    n_atom = choose_atom(lambda a: a[2].startswith("N"))
    if n_atom:
        bb_map["N"] = n_atom[1]
        n_idx = n_atom[0]
    else:
        n_idx = None

    # CA atom (type CT bonded to N)
    ca_atom = choose_atom(lambda a: a[1].upper() == "CA" or a[2].startswith("CT"))
    if n_idx and not ca_atom:
        ct_atoms = [a for a in atoms if a[2].startswith("CT")]
        bonded = [a for a in ct_atoms if n_idx in a[4]]
        if bonded:
            ca_atom = choose_atom(lambda a: a in bonded)
    if ca_atom:
        bb_map["CA"] = ca_atom[1]
        ca_idx = ca_atom[0]
    else:
        ca_idx = None

    # C atom (type C bonded to CA)
    c_atom = None
    if ca_idx:
        c_candidates = [a for a in atoms if a[2] == "C" and ca_idx in a[4]]
        if c_candidates:
            c_atom = choose_atom(lambda a: a in c_candidates)
    if not c_atom:
        c_atom = choose_atom(lambda a: a[2] == "C")
    if c_atom:
        bb_map["C"] = c_atom[1]
        c_idx = c_atom[0]
    else:
        c_idx = None

    # O atom (type O bonded to C)
    o_atom = None
    if c_idx:
        o_candidates = [a for a in atoms if a[2] == "O" and c_idx in a[4]]
        if o_candidates:
            o_atom = choose_atom(lambda a: a in o_candidates)
    if not o_atom:
        o_atom = choose_atom(lambda a: a[2] == "O")
    if o_atom:
        bb_map["O"] = o_atom[1]

    # --- Report backbone mapping ---
    missing = [x for x in ("N", "CA", "C", "O") if x not in bb_map]
    if missing:
        print(f"[WARN] Missing backbone atoms in {prepi_file}: {missing}")
        print(f"[WARN] Partial mapping found: {bb_map}")
    else:
        print(f"[ALIGN] Aligning residue {resname}: {bb_map}")

    # ---- Rewrite PDB ----
    new_lines = []
    skipped_atoms = 0
    renamed_atoms = 0

    with open(pdb_file) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")) and resname:
                pdb_res = line[17:20].strip()
                if pdb_res == resname:
                    pdb_atom = line[12:16].strip()

                    # Drop any atom not in PREPIN
                    if pdb_atom not in prepi_atom_names and pdb_atom not in bb_map:
                        skipped_atoms += 1
                        continue

                    # Rename backbone atoms
                    if pdb_atom in ("N", "CA", "C", "O") and pdb_atom in bb_map:
                        new_name = bb_map[pdb_atom].rjust(4)
                        line = line[:12] + new_name + line[16:]
                        renamed_atoms += 1

            new_lines.append(line)

    with open(out_pdb, "w") as f:
        f.writelines(new_lines)

    print(f"[OK] PDB aligned with PREPIN: {out_pdb}")
    if skipped_atoms:
        print(f"  - Removed {skipped_atoms} atoms not in PREPIN")
    if renamed_atoms:
        print(f"  - Renamed {renamed_atoms} backbone atoms")

    return out_pdb
  

def generate_tleap_input(
    output_prefix,
    sequence,
    pdb_file,
    mod_codes=None,
    residues_to_flip=None,
    nc_cycl=False,
    res_pairs=None,
    env="phil",
    out_dir="."
):
    """
    Generate a TLEaP input file for a system built directly from a modified PDB file.

    Enhancements:
    - Automatically disables peptide bonding when custom terminal residues are detected.
    - Checks that custom residue codes are loaded from prepin/frcmod files.
    - Performs charge detection and neutralization.
    - Adds per-residue N-terminal fixes and dynamic C-terminal cleanup for NC cyclization.
    """

    assert os.path.isfile(pdb_file), f"PDB file not found: {pdb_file}"

    mod_codes = list(set(mod_codes or []))
    tleap_in = os.path.join(out_dir, f"{output_prefix}_tleap.in")
    dry_in = os.path.join(out_dir, f"{output_prefix}_dry.in")

    folder_map = {
        "ptm": "fixed_ffptm",
        "nnr": "fixed_ffncaa",
        "nt": "fixed_ffnt",
        "ct": "fixed_ffct"
    }

    def write_mods(f, mods):
        loaded = set()
        for mod in mods:
            for folder in folder_map.values():
                prep = os.path.join(folder, f"{mod}.prepin")
                frcmod = os.path.join(folder, f"{mod}.frcmod")
                if os.path.isfile(prep) and prep not in loaded:
                    f.write(f"loadAmberPrep {os.path.abspath(prep)}\n")
                    loaded.add(prep)
                if os.path.isfile(frcmod) and frcmod not in loaded:
                    f.write(f"loadAmberParams {os.path.abspath(frcmod)}\n")
                    loaded.add(frcmod)
        if mods and not loaded:
            print(f"Warning: No .prepin/.frcmod files found for: {mods}")

    # -----------------------
    # DRY RUN CHARGE DETECTION
    # -----------------------
    with open(dry_in, "w") as f:
        f.write("source leaprc.protein.ff14SB\n")
        f.write("source leaprc.water.tip3p\n")
        f.write("set default PBRadii bondi\n\n")
        write_mods(f, mod_codes)
        f.write(f"mol = loadpdb {os.path.abspath(pdb_file)}\n")
        f.write("charge mol\nquit\n")

    dry = subprocess.run(
        f"tleap -f {os.path.basename(dry_in)}",
        shell=True,
        capture_output=True,
        text=True,
        cwd=out_dir
    )

    try:
        os.remove(dry_in)
    except:
        pass

    charge = None
    for line in dry.stdout.splitlines():
        if "Total unperturbed charge" in line:
            charge = float(line.split(":")[-1].strip())
    if charge is None:
        raise RuntimeError("Failed to extract total charge from dry run.")

    print(f"\nDetected total charge: {charge:+.2f}")

    # -----------------------
    # MAIN TLEAP INPUT
    # -----------------------
    with open(tleap_in, "w") as f:
        f.write("source leaprc.protein.ff14SB\n")
        f.write("source leaprc.water.tip3p\n")
        f.write("set default PBRadii bondi\n\n")

        write_mods(f, mod_codes)

        f.write(f"mol = loadpdb {os.path.abspath(pdb_file)}\n")

        # CUSTOM TERMINI
        terminal_mods = [
            m for m in mod_codes
            if m in os.listdir("fixed_ffnt") or m in os.listdir("fixed_ffct")
        ]
        if terminal_mods:
            f.write("\n# Disable auto peptide bonding for custom termini\n")
            f.write("set mol head \"\"\n")
            f.write("set mol tail \"\"\n")

        # D-RESIDUE FLIPS
        if residues_to_flip:
            f.write("\n# Flip D-residues\n")
            for idx in residues_to_flip:
                f.write(f"select mol.{idx}.CA\nflip mol\nrelax mol\ndeselect mol\n")

        # ----------------------------------
        # N-C CYCLIZATION SECTION
        # ----------------------------------
        if nc_cycl:
            # Parse sequence safely
            if isinstance(sequence, list):
                first_res = sequence[0].upper()
                last_res = sequence[-1].upper()
                num_res = len(sequence)
            else:
                seq = sequence.replace(" ", "").upper()
                first_res = seq[0:3]
                last_res = seq[-3:]
                num_res = len(seq) // 3

            f.write(f"\n# === N–C CYCLIZATION SETUP ===\n")
            f.write(f"# First residue: {first_res}\n# Last residue: {last_res}\n\n")

            # Only normalize N-term name for cysteine to avoid NCYX ➝ CYX misassignment
            if first_res == "CYX":
                f.write('set mol.1 name "CYS"\n')
                
            # --- N-terminal edits ---
            residue_blocks = {
                "GLY": """remove mol mol.1.HA2
remove mol mol.1.HA3
set mol.1.N type N
set mol.1.CA type CT
""",
                "LEU": """remove mol mol.1.HB2
remove mol mol.1.HB3
remove mol mol.1.HA
set mol.1.N type N
set mol.1.CA type CT
set mol.1.CB type CT
set mol.1.CG type CT
""",
                "ALA": """remove mol mol.1.HA
set mol.1.N type N
set mol.1.CA type CT
""",
                "CYS": """remove mol mol.1.HA
set mol.1.N type N
set mol.1.CA type CT
set mol.1.CB type CT
""",
                "ASP": """remove mol mol.1.HA
remove mol mol.1.HB2
remove mol mol.1.HB3
set mol.1.N type N
set mol.1.CA type CT
set mol.1.CB type CT
set mol.1.CG type C
""",
                "GLU": """remove mol mol.1.HB2
remove mol mol.1.HB3
remove mol mol.1.HG2
remove mol mol.1.HG3
remove mol mol.1.HA
set mol.1.N type N
set mol.1.CA type CT
set mol.1.CB type CT
set mol.1.CG type CT
set mol.1.CD type C
""",
                "PHE": """remove mol mol.1.HA
set mol.1.N type N
set mol.1.CA type CT
""",
                "HIS": """remove mol mol.1.HA
set mol.1.N type N
set mol.1.CA type CT
""",
                "ILE": """remove mol mol.1.HA
remove mol mol.1.HB
set mol.1.N type N
set mol.1.CA type CT
set mol.1.CB type CT
set mol.1.CG1 type CT
set mol.1.CG2 type CT
""",
                "LYS": """remove mol mol.1.HA
remove mol mol.1.HB2
remove mol mol.1.HB3
remove mol mol.1.HD3
set mol.1.N type N
set mol.1.CA type CT
set mol.1.CB type CT
set mol.1.CG type CT
set mol.1.CD type CT
set mol.1.CE type CT
""",
                "MET": """remove mol mol.1.HA
remove mol mol.1.HB2
remove mol mol.1.HB3
set mol.1.N type N
set mol.1.CA type CT
set mol.1.CB type CT
set mol.1.CG type CT
""",
                "ASN": """remove mol mol.1.HA
remove mol mol.1.HB2
remove mol mol.1.HB3
set mol.1.N type N
set mol.1.CA type CT
set mol.1.CB type CT
set mol.1.CG type C
set mol.1.OD1 type O
set mol.1.ND2 type N
""",
                "PRO": """remove mol mol.1.HA
remove mol mol.1.HD2
remove mol mol.1.HD3
set mol.1.N type N
set mol.1.CA type CT
""",
                "GLN": """remove mol mol.1.HA
remove mol mol.1.HB2
remove mol mol.1.HB3
set mol.1.N type N
set mol.1.CA type CT
set mol.1.CB type CT
set mol.1.CG type CT
""",
                "ARG": """remove mol mol.1.HA
remove mol mol.1.HB2
remove mol mol.1.HB3
remove mol mol.1.HD3
set mol.1.N type N
set mol.1.CA type CT
set mol.1.CB type CT
set mol.1.CG type CT
set mol.1.CD type CT
set mol.1.NE type N2
""",
                "SER": """remove mol mol.1.HA
remove mol mol.1.HB2
remove mol mol.1.HB3
remove mol mol.1.HG
set mol.1.N type N
set mol.1.CA type CT
set mol.1.CB type CT
""",
                "THR": """remove mol mol.1.HA
remove mol mol.1.HB
set mol.1.N type N
set mol.1.CA type CT
set mol.1.CB type CT
""",
                "VAL": """remove mol mol.1.HA
remove mol mol.1.HB
set mol.1.N type N
set mol.1.CA type CT
set mol.1.CB type CT
""",
                "TRP": """remove mol mol.1.HA
set mol.1.N type N
set mol.1.CA type CT
""",
                "TYR": """remove mol mol.1.HA
set mol.1.N type N
set mol.1.CA type CT
"""
            }

            if first_res in residue_blocks:
                f.write(f"# Apply N-terminal residue-specific block for {first_res}\n")
                f.write(residue_blocks[first_res])
            else:
                f.write(f"# No custom N-terminal block found for {first_res}\n")

            # --- C-terminal cleanup (generic) ---
            f.write(f"""
# Adjust C-terminal residue before cyclization
set mol.{num_res} name {last_res}
remove mol mol.{num_res}.OXT
set mol.{num_res}.C type C
set mol.{num_res}.O type O
""")

            # --- Cyclization bond ---
            f.write(f"\n# Form N–C cyclization bond\nbond mol.1.N mol.{num_res}.C\nrelax mol\n")

        # --- Disulfide bonds ---
        if res_pairs:
            f.write("\n# Disulfide bonds\n")
            for a, b in res_pairs:
                f.write(f"bond mol.{a}.SG mol.{b}.SG\nrelax mol\n")

        # --- Neutralization ---
        if charge < 0:
            f.write("\n# Neutralize negative system\naddions mol Na+ 0\n")
        elif charge > 0:
            f.write("\n# Neutralize positive system\naddions mol Cl- 0\n")

        # --- Solvation ---
        if env.lower() == "vac":
            f.write("\n# Vacuum environment: no solvation applied\n")
            
        elif env.lower() == "phil":
            f.write("\n# Hydrophilic solvation\n")
            f.write("solvateoct mol TIP3PBOX 10.0\n")

        elif env.lower() == "phob":
            f.write("\n# Hydrophobic solvation using mthbox209\n")

            # Validate hydrophobic library
            abs_lib = os.path.abspath("mthbox209.lib")
            if not os.path.isfile(abs_lib):
                raise FileNotFoundError(
                    f"Hydrophobic solvent library not found: {abs_lib}"
                )

            f.write(f"loadoff {abs_lib}\n")
            f.write("solvateBox mol mthbox209 10.0\n")

        else:
            raise ValueError(f'Unknown environment type: {env}')

        # --- Save outputs ---
        f.write(f"\n# Save outputs\nsavepdb mol {output_prefix}_tleap.pdb\n")
        f.write(f"saveamberparm mol {output_prefix}.prmtop {output_prefix}.inpcrd\n")
        f.write("quit\n")

    print(f"TLEaP input written: {tleap_in}")
    return tleap_in


def run_tleap(input_file, out_dir):
    tleap_file = os.path.basename(input_file)  # e.g., seq3_tleap.in
    print(f"Running TLEaP using {tleap_file} in {out_dir}...")
    subprocess.run(f"tleap -f {tleap_file}", shell=True, check=True, cwd=out_dir)
    print("\nTLEaP completed.")


def update_nstlim(filename, sim_time_ps):
    """
    Updates the nstlim value in the given AMBER input file based on simulation time in picoseconds.
    nstlim = sim_time_ps × 1000 (assuming 1 step = 1 fs).
    """
    nstlim = int(sim_time_ps * 1000)
    print(f"\n1ps = 1000 steps. Hence, nstlim = {nstlim}")
    
    with open(filename, 'r') as f:
        lines = f.readlines()

    new_lines = []
    nstlim_found = False

    for line in lines:
        if re.search(r'nstlim\s*=', line):
            new_line = re.sub(
                r'(nstlim\s*=\s*)\d+(,?)',
                lambda m: m.group(1) + str(nstlim) + m.group(2),
                line
            )
            new_lines.append(new_line)
            nstlim_found = True
        else:
            new_lines.append(line)

    if nstlim_found:
        with open(filename, 'w') as f:
            f.writelines(new_lines)
        print(f"{filename} successfully updated with nstlim = {nstlim}\n")
    else:
        print(f"Warning: nstlim was not found in {filename} — no changes made.")

    return nstlim


def rename_d_residues_in_pdb(pdb_file, residues_to_rename, mapping):
    if not os.path.exists(pdb_file):
        print(f"PDB file {pdb_file} not found.")
        return

    with open(pdb_file, 'r') as f:
        lines = f.readlines()

    current_res_index = 0
    last_seen_resid = None
    new_lines = []

    skip_resnames = set()

    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            resid = int(line[22:26].strip())
            resname = line[17:20].strip()

            if resid != last_seen_resid:
                if resname not in skip_resnames:
                    current_res_index += 1
                last_seen_resid = resid

            if resname not in skip_resnames and current_res_index in residues_to_rename:
                old_name = resname
                new_name = mapping.get(old_name.upper(), old_name)
                line = line[:17] + new_name.rjust(3) + line[20:]

        new_lines.append(line)

    with open(pdb_file, 'w') as f:
        f.writelines(new_lines)


def run_simulation(env, prefix, out_dir, residues_to_rename=None, mapping=None):
    print(f"Running simulations in {env} environment")

    top = f"{prefix}.prmtop"
    crd = f"{prefix}.inpcrd"
    
    MD_ENGINE = detect_mpi_engine()

    # Input files
    eminvac = os.path.abspath("eminvac.in")
    heat = os.path.abspath("heat.in")
    prodmdvac = os.path.abspath("prodmdvac.in")

    eminphil = os.path.abspath("eminphil.in")
    nvtphil = os.path.abspath("nvtphil.in")
    nptphil = os.path.abspath("nptphil.in")
    prodmdphil = os.path.abspath("prodmdphil.in")

    eminphob = os.path.abspath("eminphob.in")
    nvtphob = os.path.abspath("nvtphob.in")
    nptphob = os.path.abspath("nptphob.in")
    prodmdphob = os.path.abspath("prodmdphob.in")

    # --------------------------------------------------
    # Select simulation protocol
    # --------------------------------------------------
    if env == "vac":
        cmds = [
            ("Minimization",
             f"sander -O -i {eminvac} -o {prefix}.min -c {crd} -p {top} -r {prefix}.rst"),
            ("Heating",
             f"sander -O -i {heat} -o {prefix}.heat -c {prefix}.rst -p {top} "
             f"-r {prefix}.rst1 -x {prefix}-md1.crd -e {prefix}.ene1"),
            ("Production",
             f"sander -O -i {prodmdvac} -o {prefix}.dyn -c {prefix}.rst1 -p {top} "
             f"-r {prefix}.final -x {prefix}-md.crd -e {prefix}.ene")
        ]

    elif env == "phob":
        cmds = [
            ("Minimization",
             f"sander -O -i {eminphob} -o {prefix}.min -c {crd} -p {top} -r {prefix}.rst"),
            ("NVT equilibration",
             f"{MD_ENGINE} -O -i {nvtphob} -o {prefix}.nvt -c {prefix}.rst -p {top} "
             f"-r {prefix}.rst1 -x {prefix}-md1.crd -e {prefix}.ene1"),
            ("NPT equilibration",
             f"{MD_ENGINE} -O -i {nptphob} -o {prefix}.npt -c {prefix}.rst1 -p {top} "
             f"-r {prefix}.rst2 -x {prefix}-md2.crd -e {prefix}.ene2"),
            ("Production",
             f"{MD_ENGINE} -O -i {prodmdphob} -o {prefix}.dyn -c {prefix}.rst2 -p {top} "
             f"-r {prefix}.final -x {prefix}-md.crd -e {prefix}.ene")
        ]

    elif env == "phil":
        cmds = [
            ("Minimization",
             f"sander -O -i {eminphil} -o {prefix}.min -c {crd} -p {top} -r {prefix}.rst"),
            ("NVT equilibration",
             f"{MD_ENGINE} -O -i {nvtphil} -o {prefix}.nvt -c {prefix}.rst -p {top} "
             f"-r {prefix}.rst1 -x {prefix}-md1.crd -e {prefix}.ene1"),
            ("NPT equilibration",
             f"{MD_ENGINE} -O -i {nptphil} -o {prefix}.npt -c {prefix}.rst1 -p {top} "
             f"-r {prefix}.rst2 -x {prefix}-md2.crd -e {prefix}.ene2"),
            ("Production",
             f"{MD_ENGINE} -O -i {prodmdphil} -o {prefix}.dyn -c {prefix}.rst2 -p {top} "
             f"-r {prefix}.final -x {prefix}-md.crd -e {prefix}.ene")
        ]

    else:
        raise ValueError(f"Unknown environment: {env}")

    # --------------------------------------------------
    # Run simulations
    # --------------------------------------------------
    for stage, cmd in cmds:
        print(f"\nStarting {stage}")
        print(f"Executing command: {cmd}")

        result = subprocess.run(cmd, shell=True, cwd=out_dir)
        if result.returncode != 0:
            raise RuntimeError(
                f"Execution failed during {stage} due to bad constraints/SHAKE error."
            )

        # === Convert minimized structure immediately ===
        if stage == "Minimization":
            subprocess.run(
                f"ambpdb -p {top} -c {prefix}.rst > {prefix}_min.pdb",
                shell=True,
                cwd=out_dir,
                check=True
            )

    # --------------------------------------------------
    # Convert final (MD) structure
    # --------------------------------------------------
    subprocess.run(
        f"ambpdb -p {top} -c {prefix}.final > {prefix}_final.pdb",
        shell=True,
        cwd=out_dir,
        check=True
    )

    # --------------------------------------------------
    # Rename D-residues (min + final)
    # --------------------------------------------------
    for tag in ["min", "final"]:
        amb_pdb = Path(out_dir) / f"{prefix}_{tag}.pdb"
        if amb_pdb.exists() and residues_to_rename and mapping:
            rename_d_residues_in_pdb(str(amb_pdb), residues_to_rename, mapping)

    print("\nSimulations completed.")


def parse_energy_line(line):
    vals = {}
    parts = line.split()
    i = 0
    while i < len(parts) - 2:
        if parts[i+1] == '=':
            key = parts[i]
            val = parts[i+2]
            vals[key] = val
            i += 3
        else:
            i += 1
    return vals


def plot_energy(prefix, out_dir):
    etot, ek, ep, steps = [], [], [], []
    dyn_file = os.path.join(out_dir, f"{prefix}.dyn")
    if not os.path.exists(dyn_file):
        print(f"Warning: {dyn_file} not found.")
        return

    current_step = None
    with open(dyn_file) as f:
        for line in f:
            if "NSTEP" in line:
                vals = parse_energy_line(line)
                try:
                    current_step = int(vals.get("NSTEP"))
                except (TypeError, ValueError):
                    current_step = None
            elif "Etot" in line and current_step is not None:
                vals = parse_energy_line(line)
                try:
                    etot_val = float(vals.get("Etot"))
                    ek_val = float(vals.get("EKtot"))
                    ep_val = float(vals.get("EPtot"))
                    steps.append(current_step)
                    etot.append(etot_val)
                    ek.append(ek_val)
                    ep.append(ep_val)
                except (TypeError, ValueError):
                    print(f"Skipping line (parse error): {line.strip()}")
                current_step = None  # reset

    if steps:
        plt.figure(figsize=(8, 5))
        plt.plot(steps, etot, label="Etot")
        plt.plot(steps, ek, label="EKtot")
        plt.plot(steps, ep, label="EPtot")
        plt.legend()
        plt.xlabel("Step")
        plt.ylabel("Energy (kcal/mol)")
        plt.title("Energy vs Step")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{prefix}_energy_plot.png"))
        plt.close()
        print("Saved energy_plot.png")
    else:
        print("No valid energy data found. Plot not generated.")


def plot_rmsd(prefix, out_dir):
    trajrms_path = os.path.join(out_dir, "trajrms.in")
    rmsd_dat_path = os.path.join(out_dir, f"{prefix}_rmsd.dat")
    rmsd_plot_path = os.path.join(out_dir, f"{prefix}_rmsd_plot.png")

    with open(trajrms_path, "w") as f:
        f.write(f"""parm {prefix}.prmtop
trajin {prefix}-md.crd
reference lowestene.pdb
rms reference out {os.path.basename(rmsd_dat_path)} @N,CA,C time 1.0
run
""")

    res = subprocess.run(["cpptraj", "-i", os.path.basename(trajrms_path)], capture_output=True, text=True, cwd=out_dir)
    print(res.stdout)

    if res.returncode != 0:
        print(res.stderr)
        return

    if not os.path.exists(rmsd_dat_path):
        print(f"RMSD output not found: {rmsd_dat_path}")
        return

    data = np.loadtxt(rmsd_dat_path)
    plt.figure()
    plt.plot(data[:,0], data[:,1], color="red")
    plt.xlabel("Frame")
    plt.ylabel("RMSD (Å)")
    plt.title("RMSD vs Frame")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(rmsd_plot_path)
    plt.close()
    print(f"Saved {rmsd_plot_path}")


def run_analysis(out_dir, output_prefix, num_residues, residues_to_rename=None, mapping=None, do_traj=False, do_cluster=False, do_graph=False):
    # === TRAJECTORY EXTRACTION ===
    if do_traj:
        traj_in = os.path.join(out_dir, f"{output_prefix}_trajfile.in")
        lowestene_pdb = os.path.join(out_dir, "lowestene.pdb")

        with open(traj_in, "w") as f:
            f.write(f"""parm {output_prefix}.prmtop
trajin {output_prefix}-md.crd
trajout lowestene.pdb pdb
""")
        print(f"{traj_in} written.\n")

        res = subprocess.run(["cpptraj", "-i", os.path.basename(traj_in)], capture_output=True, text=True, cwd=out_dir)
        print(res.stdout)
        if res.returncode != 0:
            print("\nError extracting frame:", res.stderr)
            exit(1)
        print("lowestene.pdb generated\n")

        if os.path.exists(lowestene_pdb) and residues_to_rename and mapping:
            rename_d_residues_in_pdb(lowestene_pdb, residues_to_rename, mapping)
    else:
        print("\nTrajectory extraction skipped.\n")

    # === CLUSTER ANALYSIS ===
    if do_cluster:
        clust_dir = os.path.join(out_dir, "clust")
        os.makedirs(clust_dir, exist_ok=True)

        cluster_in = os.path.join(out_dir, f"{output_prefix}_cf.in")

        with open(cluster_in, "w") as f:
            f.write(f"""parm {output_prefix}.prmtop
trajin {output_prefix}-md.crd
autoimage
strip :WAT
trajout clust/traj.pdb pdb
cluster C0 dbscan minpoints 5 epsilon 2.0 sieve 10 rms :1-{num_residues}@C,N,O,CA nofit \\
out clust/cluster_num_vs_time.dat \\
summary clust/centroids.stats \\
info clust/centroids \\
cpopvtime clust/cluster_pop.agr \\
repout clust/cluster repfmt pdb \\
avgout clust/cluster_avg avgfmt pdb
run
""")
        print(f"{cluster_in} written.\n")

        res = subprocess.run(["cpptraj", "-i", os.path.basename(cluster_in)], capture_output=True, text=True, cwd=out_dir)
        if res.returncode != 0:
            print("\nError in cluster analysis:", res.stderr)
            exit(1)
        print("Clustering complete.\n")

        for fname in os.listdir(clust_dir):
            if fname.endswith(".pdb"):
                path = os.path.join(clust_dir, fname)
                if os.path.exists(path) and residues_to_rename and mapping:
                    rename_d_residues_in_pdb(path, residues_to_rename, mapping)
        print("Clustered PDB files updated with D-residue names.\n")

    else:
        print("Cluster analysis skipped.\n")

    # === ENERGY + RMSD PLOT ===
    if do_graph:
        plot_energy(output_prefix, out_dir)
        lowestene_pdb = os.path.join(out_dir, "lowestene.pdb")
        if do_traj and os.path.exists(lowestene_pdb):
            plot_rmsd(output_prefix, out_dir)
        else:
            print("RMSD plot skipped: lowestene.pdb not available.\n")
    else:
        print("Energy & RMSD graphing skipped.\n")


def compare_chirality(prefix, residues_to_flip, residues_to_flip_tleap, out_dir):
    """
    Compare chirality between L- and D-form residues in predicted vs tleap PDBs.
    Only checks the main chiral center (Cα) per residue.
    """

    l_pdb = os.path.join(out_dir, f"{prefix}_pred.pdb")
    d_pdb = os.path.join(out_dir, f"{prefix}_tleap.pdb")

    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, f"{prefix}_chirality_report.txt")

    print("\n--- CHIRALITY CHECK ---")
    with open(report_path, "w") as report:
        report.write("\n--- CHIRALITY CHECK ---\n")

        # === Helper: Identify main chiral carbon (Cα) ===
        def get_main_chiral_center(mol):
            chiral_centers = Chem.FindMolChiralCenters(mol, force=True)
            for idx, cfg in chiral_centers:
                atom = mol.GetAtomWithIdx(idx)
                neighbor_elems = {n.GetSymbol() for n in atom.GetNeighbors()}
                # Heuristic: Cα is a carbon bonded to both N and C atoms
                if atom.GetSymbol() == "C" and {"N", "C"} <= neighbor_elems:
                    return idx, cfg
            # Fallback to first chiral center if none match heuristic
            return chiral_centers[0] if chiral_centers else (None, None)

        for l_id, d_id in zip(residues_to_flip, residues_to_flip_tleap):
            # Extract residue molecules
            l_mol, l_name, _ = extract_residue_mol(l_pdb, l_id)
            d_mol, d_name, _ = extract_residue_mol(d_pdb, d_id)

            print(f"\nResidue {l_name} (L) at {l_id} in original sequence vs {d_name} (D) at {d_id} in the D-form sequence:\n")
            report.write(f"\nResidue {l_name} (L) at {l_id} in original sequence vs {d_name} (D) at {d_id} in the D-form sequence:\n")

            # Handle extraction errors
            if not l_mol or not d_mol:
                msg = "  - Could not extract one or both residues."
                print(msg)
                report.write(msg + "\n")
                continue

            # Find main chiral center for both
            l_idx, l_cfg = get_main_chiral_center(l_mol)
            d_idx, d_cfg = get_main_chiral_center(d_mol)

            if l_idx is None or d_idx is None:
                msg = "  - No chiral center detected in one or both residues."
                print(msg)
                report.write(msg + "\n")
                continue

            # Compare configurations
            status = "INVERTED" if l_cfg != d_cfg else "IDENTICAL (NO INVERSION)"
            atom_symbol = l_mol.GetAtomWithIdx(l_idx).GetSymbol()

            # Output to console and report
            print(f"  - L-form → Name: {l_name}, Residue ID: {l_id}")
            print(f"  - D-form → Name: {d_name}, Residue ID: {d_id}")
            print(f"  - Chirality: Atom {l_idx} ({atom_symbol}) → L-form: {l_cfg}, D-form: {d_cfg} -> {status}\n")

            report.write(f"  - L-form → Name: {l_name}, Residue ID: {l_id}\n")
            report.write(f"  - D-form → Name: {d_name}, Residue ID: {d_id}\n")
            report.write(f"  - Chirality: Atom {l_idx} ({atom_symbol}) → L-form: {l_cfg}, D-form: {d_cfg} -> {status}\n")

    print(f"\nChirality report saved to {report_path}")
    

def extract_residue_mol(pdb_path, res_id):
    pdb_block = ""
    resname = None
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") and int(line[22:26].strip()) == res_id:
                pdb_block += line
                if not resname:
                    resname = line[17:20].strip()

    if not pdb_block:
        return None, None, res_id  # Return full 3-tuple even if failed

    mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    if mol:
        try:
            Chem.SanitizeMol(mol)
            AllChem.AssignAtomChiralTagsFromStructure(mol)
        except:
            return None, resname, res_id
    return mol, resname, res_id  # full tuple


def log_distance(message, distance, out_file):
    with open(out_file, "a") as f:
        if distance is None:
            f.write(f"{message}: atoms not found (cyclized)\n")
        else:
            f.write(f"{message}: {distance:.2f} Å\n")

    
def prepare_single_sequence(seq, prefix, args, out_dir, before_sim_dir):
    """
    PREPARATION PHASE
    Generates clean.pdb and tleap.pdb ONLY
    """

    # === 1. Parse sequence ===
    seq, modifications, n_cap, c_cap, nc_cycl, res_pairs = parse_sequence(seq)
    validate_sequence_letters(seq)

    # === 2. Predict structure ===
    pred_pdb = predict_structure(
        seq,
        out_dir=out_dir,
        jobname=f"{prefix}_pred",
        method=args.method
    )

    clean_pdb_file = os.path.join(out_dir, f"{prefix}_clean.pdb")
    clean_pdb(pred_pdb, clean_pdb_file)

    # === 3. Apply modifications ===
    replacements = {
        m["position"]: m["modification"]
        for m in modifications
        if m["mod_type"] in {"ptm", "nnr", "nnm"}
    }

    first_residue = seq[0]
    last_residue = seq[-1]

    nterm_mod_final = resolve_terminal_mod(n_cap, first_residue, "nt") if n_cap else None
    cterm_mod_final = resolve_terminal_mod(c_cap, last_residue, "ct") if c_cap else None

    pdb_modified = os.path.join(out_dir, f"{prefix}_modified.pdb")
    modify_pdb(
        pdb_file=clean_pdb_file,
        replacements=replacements,
        output_file=pdb_modified,
        nterm_mod=nterm_mod_final,
        cterm_mod=cterm_mod_final,
    )

    # === 3b. Align atoms ===
    aligned_pdb = pdb_modified
    all_mods = [m["modification"] for m in modifications if m["mod_type"] in {"ptm", "nnr"}]
    if nterm_mod_final:
        all_mods.append(nterm_mod_final)
    if cterm_mod_final:
        all_mods.append(cterm_mod_final)

    for mod in set(all_mods):
        for folder in ["fixed_ffptm", "fixed_ffncaa", "fixed_ffnt", "fixed_ffct"]:
            prepi = os.path.join(folder, f"{mod}.prepin")
            if os.path.isfile(prepi):
                aligned_tmp = os.path.join(out_dir, f"{prefix}_{mod}_aligned.pdb")
                aligned_pdb = align_pdb_atoms_to_prepi(
                    aligned_pdb, prepi, aligned_tmp
                )
                break

    # === 4. Disulfides ===
    pdb_for_tleap = aligned_pdb
    if res_pairs:
        cyx_pdb = os.path.join(out_dir, f"{prefix}_cyx.pdb")
        rename_cys_to_cyx_and_add_conect(aligned_pdb, res_pairs, cyx_pdb)
        pdb_for_tleap = cyx_pdb

    # === 5. Extract sequence ===
    raw_seq = extract_sequence_from_pdb(pdb_for_tleap)

    residues_to_flip = [m["position"] for m in modifications if m["mod_type"] == "d"]

    # === 6. Run TLEaP ===
    tleap_in = generate_tleap_input(
        output_prefix=prefix,
        sequence=raw_seq,
        pdb_file=pdb_for_tleap,
        mod_codes=list(set(all_mods)),
        residues_to_flip=residues_to_flip,
        nc_cycl=nc_cycl,
        res_pairs=res_pairs,
        env=args.env,
        out_dir=out_dir,
    )

    run_tleap(tleap_in, out_dir=out_dir)

    # === COPY ONLY tleap.pdb to GLOBAL before_simulation ===
    tleap_pdb = os.path.join(out_dir, f"{prefix}_tleap.pdb")
    shutil.copy(
        tleap_pdb,
        os.path.join(before_sim_dir, f"{prefix}_tleap.pdb")
    )

    return {
        "prefix": prefix,
        "out_dir": out_dir,
        "seq_len": len(raw_seq),
        "residues_to_flip": residues_to_flip,
        "nc_cycl": nc_cycl,
        "res_pairs": res_pairs,
    }


def run_single_simulation(meta, args):
    prefix = meta["prefix"]
    out_dir = meta["out_dir"]
    residues_to_flip = meta["residues_to_flip"]
    residues_to_flip_tleap = residues_to_flip[:]

    update_nstlim(f"prodmd{args.env}.in", args.sim)

    run_simulation(
        args.env,
        prefix,
        out_dir,
        residues_to_rename=residues_to_flip,
        mapping=D_RESI_MAP
    )

    run_analysis(
        out_dir,
        prefix,
        meta["seq_len"],
        residues_to_rename=residues_to_flip,
        mapping=D_RESI_MAP,
        do_traj=(args.traj == "yes"),
        do_cluster=(args.cluster == "yes"),
        do_graph=(args.graph == "yes"),
    )

    # Chirality check
    if residues_to_flip:
        compare_chirality(prefix, residues_to_flip, residues_to_flip_tleap, out_dir)

    # Distance checks
    clean_pdb = os.path.join(out_dir, f"{prefix}_clean.pdb")
    final_pdb = os.path.join(out_dir, f"{prefix}_final.pdb")
    dist_report = os.path.join(out_dir, f"{prefix}_distance_report.txt")

    if meta.get("nc_cycl"):
        d1 = get_nc_distance(clean_pdb)
        d2 = get_nc_distance(final_pdb)
        log_distance("Before cyclization", d1, dist_report)
        log_distance("After cyclization", d2, dist_report)

    if meta.get("res_pairs"):
        print_sg_distances(clean_pdb, meta["res_pairs"], "Before cyclization", dist_report)
        print_sg_distances(final_pdb, meta["res_pairs"], "After cyclization", dist_report)


def main():
    parser = argparse.ArgumentParser(
        description="Chemical modifications + AMBER + ESMFold/alphafold2 pipeline"
    )
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-p", "--prefix", default="test")
    parser.add_argument("-o", "--outdir", default="output")
    parser.add_argument("-e", "--env", choices=["vac", "phob", "phil"], default="phil")
    parser.add_argument("-s", "--sim", type=float, default=100.0)
    parser.add_argument("-t", "--traj", choices=["yes", "no"], default="no")
    parser.add_argument("-c", "--cluster", choices=["yes", "no"], default="no")
    parser.add_argument("-g", "--graph", choices=["yes", "no"], default="no")
    parser.add_argument(
        "-m", "--method", choices=["esmfold", "alphafold2"], default="alphafold2"
    )
    args = parser.parse_args()

    named_sequences = read_named_sequences(args.input, args.prefix)
    os.makedirs(args.outdir, exist_ok=True)

    # ==================================================
    # GLOBAL FOLDERS
    # ==================================================
    before_sim_dir = os.path.join(args.outdir, "before_simulation")
    after_min_dir = os.path.join(args.outdir, "after_minimization")
    after_sim_dir = os.path.join(args.outdir, "after_simulation")

    os.makedirs(before_sim_dir, exist_ok=True)
    os.makedirs(after_min_dir, exist_ok=True)
    os.makedirs(after_sim_dir, exist_ok=True)

    log_file = os.path.join(args.outdir, f"{args.prefix}_error.log")
    prep_queue = []

    # ==================================================
    # PHASE 1: STRUCTURE PREPARATION
    # ==================================================
    print("\nPHASE 1 STRUCTURE PREPARATION\n")

    with open(log_file, "w") as elog:
        for i, (name, seq) in enumerate(named_sequences, 1):
            seq_outdir = os.path.join(args.outdir, name)
            os.makedirs(seq_outdir, exist_ok=True)
            print(f"PREP {i}/{len(named_sequences)} {name}")
            try:
                meta = prepare_single_sequence(
                    seq, name, args, seq_outdir, before_sim_dir
                )
                prep_queue.append(meta)
            except Exception as e:
                msg = f"PREP ERROR {i} {name} {e}"
                print(msg)
                elog.write(msg + "\n")

    # ==================================================
    # PHASE 2: SIMULATIONS
    # ==================================================
    print("\nPHASE 2 SIMULATIONS\n")

    with open(log_file, "a") as elog:
        for i, meta in enumerate(prep_queue, 1):
            prefix = meta["prefix"]
            print(f"SIM {i}/{len(prep_queue)} {prefix}")
            try:
                run_single_simulation(meta, args)
            except Exception as e:
                msg = f"SIM ERROR {i} {prefix} {e}"
                print(msg)
                elog.write(msg + "\n")

    # ==================================================
    # COLLECT MINIMIZED STRUCTURES (PEPstrMOD BENCHMARK)
    # ==================================================
    print("\nCollecting minimized PDB files")
    for name, _ in named_sequences:
        src = os.path.join(args.outdir, name, f"{name}_min.pdb")
        dst = os.path.join(after_min_dir, f"{name}_min.pdb")
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Skipped {src}")

    # ==================================================
    # COLLECT FINAL (MD) STRUCTURES
    # ==================================================
    print("\nCollecting final PDB files")
    for name, _ in named_sequences:
        src = os.path.join(args.outdir, name, f"{name}_final.pdb")
        dst = os.path.join(after_sim_dir, f"{name}_final.pdb")
        if os.path.exists(src):
            shutil.copy(src, dst)
        else:
            print(f"Skipped {src}")

    # ==================================================
    # ZIP OUTPUTS
    # ==================================================
    after_min_zip = os.path.join(args.outdir, "after_minimization.zip")
    with zipfile.ZipFile(after_min_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(after_min_dir):
            zipf.write(
                os.path.join(after_min_dir, file),
                arcname=os.path.join("after_minimization", file)
            )
    print(f"Minimized PDBs zipped to {after_min_zip}")

    after_sim_zip = os.path.join(args.outdir, "after_simulation.zip")
    with zipfile.ZipFile(after_sim_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(after_sim_dir):
            zipf.write(
                os.path.join(after_sim_dir, file),
                arcname=os.path.join("after_simulation", file)
            )
    print(f"Final PDBs (Simulated PDBs) zipped to {after_sim_zip}")

    before_zip = os.path.join(args.outdir, "before_simulation.zip")
    with zipfile.ZipFile(before_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(before_sim_dir):
            zipf.write(
                os.path.join(before_sim_dir, file),
                arcname=os.path.join("before_simulation", file)
            )
    print(f"Preparation (Initial structure) PDBs zipped to {before_zip}")


if __name__ == "__main__":
    main()
