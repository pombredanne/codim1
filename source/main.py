# Generate the mesh

# node_pos contains the position of each node in tuple form (x, y)
node_pos = np.zeros(11, 1)
for i in range(0, 11):
    node_pos.append((float(i), 0.0))

# Element to node contains pairs of indices referring the (x, y) values in
# node_pos
element_to_node = []
for i in range(0, 10):
    element_to_node.append((i, i + 1))

element_centroid = [
    (0.5 * (node_pos[e[0]][0] + node_pos[e[1]][0]),
     0.5 * (node_pos[e[0]][1] + node_pos[e[1]][1]))
    for e in element_to_node]


