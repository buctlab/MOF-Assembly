import numpy as np
import re
import time
from ase import Atoms
from ase.io import read
from ase.visualize.plot import plot_atoms
import matplotlib.pyplot as plt
import sys
import random

distance_threshold = 1.0

vertices = ('H' , 'He', 'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar',
	  'K' , 'Ca', 'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
	  'Rb', 'Sr', 'Y' , 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I' , 'Xe',
	  'Cs', 'Ba', 'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr',
	  'Ra', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Ac', 'Th',
	  'Pa', 'U' , 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'FG', 'X' )

def isfloat(value):
	try:
		float(value)
		return True
	except ValueError:
		return False

def nn(string):
	return re.sub('[^a-zA-Z]','', string)

def isvert(line):
    if len(line) >= 5:
        if nn(line[0]) in vertices and line[1] in vertices and False not in map(isfloat, line[2:5]):
            return True
        else:
            return False

def read_cif_file(cif_file_path):

    with open(cif_file_path, 'r') as f:
        lines = f.readlines()
    node_coord_list = []
    node_index_list = []
    for line in lines:
        s = line.split()
        if isvert(s):
            node_coord_list.append([s[0],float(s[2]),float(s[3]),float(s[4])])
            node_index_list.append(s[0])

    return node_coord_list,node_index_list

def distance(p1, p2):

    return np.sqrt((p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2 + (p1[3] - p2[3]) ** 2)

def find_closest_pair(points, k, iterations):

    if len(points) < 2 or k < 2:
        return None, None, float('inf')

    current_point = random.choice(points)
    points.remove(current_point)

    mindis = float('inf')
    closest_pair = (current_point, None)

    for _ in range(iterations):

        if k > len(points):
            candidate_points = points
        else:
            candidate_points = random.sample(points, k)

        for point in candidate_points:
            d = distance(current_point, point)
            if d < mindis:
                mindis = d
                closest_point = point
                closest_pair = (current_point, closest_point)

        current_point = closest_point
        if current_point in points:
            points.remove(current_point)

    return  mindis,closest_pair

if __name__ == '__main__':

    node_coord_list, node_index_list = read_cif_file('./testdataset/afw_v1-4c_In_1_Ch_1-1B_1TrU.cif')
    atoms = read('./testdataset/afw_v1-4c_In_1_Ch_1-1B_1TrU.cif')
    print("The total number of atoms is " + len(node_coord_list))

    for node,atom in zip(node_coord_list,atoms):
        node[1] = atom.position[0]
        node[2] = atom.position[1]
        node[3] = atom.position[2]
    print(node_coord_list)

    start_time = time.time()
    flag = False

    k = 50
    iterations = 1104
    min_dist,points= find_closest_pair(node_coord_list,k, iterations)
    if min_dist < distance_threshold:
        flag = True
        print(points)
    end_time = time.time()

    if not flag:
        print("There are no two atoms whose distance is less than a given threshold.")

    if flag:
        print("There exist two atoms whose distance is less than the given threshold!")
        print("The minimum distance between atoms is ", min_dist)

    index1 = node_index_list.index(points[0][0])
    index2 = node_index_list.index(points[1][0])
    print(index1,index2)
    run_time = end_time - start_time
    print("The running time of the program is:",run_time)