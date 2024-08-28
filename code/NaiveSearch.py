from ase.io import read
import re
import time
import numpy as np

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

if __name__ == '__main__':

    node_coord_list,node_index_list = read_cif_file('./testdataset/afw_v1-4c_In_1_Ch_1-1B_1TrU.cif')
    atoms = read('./testdataset/afw_v1-4c_In_1_Ch_1-1B_1TrU.cif')
    print("The total number of atoms is " + len(node_coord_list))

    for node,atom in zip(node_coord_list,atoms):
        node[1] = atom.position[0]
        node[2] = atom.position[1]
        node[3] = atom.position[2]

    start_time = time.time()
    flag = False
    min_dist = float('inf')

    for i in range(len(node_coord_list)):
        point1 = node_coord_list[i][0]
        coords1 = np.array(node_coord_list[i][1:4])
        for j in range(i+1, len(node_coord_list)):
            point2 = node_coord_list[j][0]
            coords2 = np.array(node_coord_list[j][1:4])
            dist = np.linalg.norm(coords1 - coords2)
            if dist < min_dist:
                min_dist = dist
                points = (point1, point2)

    if min_dist < distance_threshold:
        flag = True
        print(points)
    end_time = time.time()

    if not flag:
        print("There are no two atoms whose distance is less than a given threshold.")

    if flag:
        print("There exist two atoms whose distance is less than the given threshold!")
        print("The minimum distance between atoms is ", min_dist)

    index1 = node_index_list.index(points[0])
    index2 = node_index_list.index(points[1])
    print(index1,index2)
    run_time = end_time - start_time
    print("The running time of the program is:",run_time)