## Import packages
import numpy as np
from numpy import sin, cos, sqrt, pi
from ase import Atoms
from os import listdir
from os.path import isfile, isdir, join, splitext

## Given a path of directory, output the list of file names with constrained filename extension.
## For example, extension = '.cif'
def get_file_list_with_extension(folder_path, extension):
    # Get all names of files and dirctories under the target folder.
    files = listdir(folder_path)
    # Find the files with the desired extension
    constrained_files = []
    for str in files:
        extension_name = splitext(folder_path + '/' + str)[1]
        if extension_name == extension:
            constrained_files.append(str)
    # Using sort() function to sort the list of strings
    constrained_files.sort()
    return constrained_files

## The input elements scales and angles can be defined as follows:
## scales = np.array([a, b, c], dtype = 'float')
## angles = np.array([alp, bet, gam], dtype = 'float')
def get_unit_cell(scales, angles):
    ## In python, elements separated by ',' can recieve information from np.array
    ## Set scales (a, b, c) and angles (alp, bet, gam)
    a,b,c = scales
    ## alpha, beta, gamma
    alp, bet, gam = angles/360*2*pi
    ## In python the notation ** denotes the Exponentiation, e.g.,	x ** y
    Omg = a*b*c*sqrt(1-cos(alp)**2-cos(bet)**2-cos(gam)**2+2*cos(alp)*cos(bet)*cos(gam))
    ##　"Row" vectors: v1, v2, v3
    v1 = np.array([a, 0, 0])
    v2 = np.array([b*cos(gam), b*sin(gam), 0])
    v3 = np.array([c*cos(bet), c*(cos(alp)-cos(bet)*cos(gam))/sin(gam), Omg/(a*b*sin(gam))])
    ## The unit_cell consists of "row" vectors v1, v2, and v3
    unit_cell = np.array([v1, v2, v3])
    return unit_cell

## A function for reading cif files without using 'gemmi'
## The purpose of this functions is as follows:
## 1. Get scales and angles information with key values 'scales' and 'angles'
## 2. Use function get_unit_cell() to compute the unit cell of the cif file
## 3. Get the atom types and coordinates with key values 'atoms' and 'atom2coords'
## 4. Get the dictionary of properties with the key value 'properties'
def read_cif(path2cif):
    ## Step 1. Read the cif file (only use the function open())
    f = open(path2cif)
    unit_cell_raw = dict()
    unit_cell_keys = ['_cell_length_a',
                      '_cell_length_b',
                      '_cell_length_c',
                      '_cell_angle_alpha',
                      '_cell_angle_beta',
                      '_cell_angle_gamma']
    for l in f.readlines():
        if len(l.split())>0 and (l.split()[0] in unit_cell_keys):
            key, value = l.split()
            unit_cell_raw[(key.strip().split('_'))[-1]] = float(value.strip())
            #print(l.split()[0], l.split()[1])
    f.close()
    ## Step 2. Obtain the scales and angles from unit_cell_raw derived in Step 1
    a, b, c = unit_cell_raw['a'], unit_cell_raw['b'], unit_cell_raw['c']
    alp, bet, gam = unit_cell_raw['alpha'], unit_cell_raw['beta'], unit_cell_raw['gamma']
    ## Step 3. Use function get_unit_cell() to obtain the matrix of the unit cell
    scales = np.array([a, b, c], dtype = 'float')
    angles = np.array([alp, bet, gam], dtype = 'float')
    unit_cell = get_unit_cell(scales, angles)
    ## Step 4. Get the atom types and atom coordinates
    atom2coords = dict()
    f = open(path2cif)
    for l in f.readlines():
        items = l.split()
        if len(items)>2 and (items[0].startswith(items[1])):
            atom = items[1]
            coords = [float(items[i]) for i in range(2, 5)]
            if atom in atom2coords.keys():
                atom2coords[atom] = np.concatenate((atom2coords[atom], np.array(coords).reshape(1,3)))
            else:
                atom2coords[atom] = np.array(coords).reshape(1,3)
    # The atom2coords is a dictionary, so the atom types (atoms) are the keys of atom2coords
    atoms = list(atom2coords.keys())
    ## Step 5. Get the properties in the CIF file
    f = open(path2cif)
    # The properties is a dictionary. It consists of the name of perperties and corresponding values
    properties = dict()
    for l in f.readlines():
        if l.startswith('# '):
            key_value = l.split(':')
            key = key_value[0][2:].strip()
            value = key_value[1].strip()
            properties[key] = value
    f.close()
    ## Step 6. Finally, collect the obtained data info as a dictionary
    cif_data = {'scales': scales,
                'angles': angles,
                'unit_cell': unit_cell,
                'atoms': atoms,
                'atom2coords': atom2coords,
                'properties': properties,
               }
    return cif_data

## Given a path of directory of cif files, output the list of all atom types occur in the files.
def get_all_atom_types(folder_path):
    constrained_files = get_file_list_with_extension(folder_path, '.cif')
    ## Notice that here we use "set" to collect all atom types
    atom_distinct = set()
    for cif_name in constrained_files:
        path = folder_path+'/{}'.format(cif_name)
        cif_data = read_cif(path)
        atoms = cif_data['atoms']
        for atom in atoms:
            atom_distinct.add(atom)
    return atom_distinct
