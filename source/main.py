import numpy as np

n_elements = 10

# Generate the mesh
# node_pos contains the position of each node in tuple form (x, y)
n_nodes = n_elements + 1
node_pos = np.zeros(n_nodes)
for i in range(0, n_nodes):
    node_pos.append((float(i), 0.0))

# Element to node contains pairs of indices referring the (x, y) values in
# node_pos
element_to_node = []
for i in range(0, n_elements):
    element_to_node.append((i, i + 1))

# Calculate the center of each element
element_centroid = [
    (0.5 * (node_pos[e[0]][0] + node_pos[e[1]][0]),
     0.5 * (node_pos[e[0]][1] + node_pos[e[1]][1]))
    for e in element_to_node]

# Set input stresses
sigma_n = np.ones(n_elements)
sigma_s = np.zeros(n_elements)



