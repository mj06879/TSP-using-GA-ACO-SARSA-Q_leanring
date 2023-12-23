import numpy as np

def read_tsp_file(file_path):
    coordinates = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('EOF'):
                break
            parts = line.split()
            if parts[0].isdigit():
                coordinates.append((float(parts[1]), float(parts[2])))
    return coordinates

def compute_distance_matrix(coordinates):
    n = len(coordinates)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[j]
            distance_matrix[i, j] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return distance_matrix

def save_distance_matrix(distance_matrix, output_file):
    np.savetxt(output_file, distance_matrix, fmt='%.2f', delimiter='\t')

def dist_matrix(tsp_file_path):
    coordinates = read_tsp_file(tsp_file_path)
    distance_matrix = compute_distance_matrix(coordinates)
    # save_distance_matrix(distance_matrix, output_txt_file)
    # print(type(distance_matrix))
    return distance_matrix

# Example usage:
# tsp_file_path = 'qa194.tsp'
# output_txt_file = 'distance_matrix.txt'

# coordinates = read_tsp_file(tsp_file_path)
# distance_matrix = compute_distance_matrix(coordinates)
# save_distance_matrix(distance_matrix, output_txt_file)

# dist_matrix('data/qa194.tsp', 'distance_matrix.txt')
# print(type(np.loadtxt('distance_matrix.txt')))
