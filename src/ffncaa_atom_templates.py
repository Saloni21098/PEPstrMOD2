from Bio.PDB import PDBParser
import os
import re
from available_ncaas import MODCODE_LOOKUP


def load_template_atoms(template_dir):
    parser = PDBParser(QUIET=True)
    natural_atoms = {}

    for fname in os.listdir(template_dir):
        if fname.endswith(".pdb"):
            resname = fname.split(".")[0].upper()  # E.g., "A"
            path = os.path.join(template_dir, fname)
            structure = parser.get_structure(resname, path)

            for model in structure:
                for chain in model:
                    for residue in chain:
                        atoms = [a.get_name() for a in residue]
                        natural_atoms[resname] = atoms
                        break  # Only first residue
                    break
                break

    return natural_atoms


def parse_ffncaa(ffncaa_folder, template_dir):
    allowed_atoms = {}
    rename_map = {}

    # Load natural templates
    natural_templates = load_template_atoms(template_dir)

    # Loop over all .prepin files
    for fname in sorted(os.listdir(ffncaa_folder)):
        if not fname.endswith(".prepin"):
            continue
        prepin_path = os.path.join(ffncaa_folder, fname)
        current_res = None
        atoms = []

        with open(prepin_path) as f:
            for line in f:
                if not line.strip():
                    continue

                # Detect residue start
                if "INT" in line and len(line.strip().split()) >= 2:
                    if current_res and atoms:
                        allowed_atoms[current_res] = atoms
                    current_res = line.strip().split()[0]  # e.g., "AIB"
                    atoms = []

                # Detect atom lines
                elif re.match(r"\s*\d+\s+\w+\s+\w+\s+\w+", line):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        atom_name = parts[1]
                        atoms.append(atom_name)

            if current_res and atoms:
                allowed_atoms[current_res] = atoms

    # Compare with base residue atoms
    for mod_code, mod_atoms in allowed_atoms.items():
        for (base_res, mod) in MODCODE_LOOKUP:
            if mod == mod_code:
                nat_atoms = natural_templates.get(base_res)
                if not nat_atoms:
                    print(f"Warning: No template for base residue {base_res}")
                    continue

                for nat_atom in nat_atoms:
                    if nat_atom not in mod_atoms:
                        # Try to find a substitute
                        for mod_atom in mod_atoms:
                            if mod_atom.startswith(nat_atom):
                                rename_map.setdefault(mod_code, {})[nat_atom] = mod_atom
                                break

    return allowed_atoms, rename_map

