import pickle
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import os


def get_weighted_adjacency_matrix(distance_df, sensor_ids):
    """

    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :return:
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind or len(row) != 3:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]
        dist_mx[sensor_id_to_ind[row[1]], sensor_id_to_ind[row[0]]] = row[2]
    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()

    adj_mx = np.exp(-np.square(dist_mx / std))

    return sensor_ids, sensor_id_to_ind, adj_mx


def get_time_volume_matrix(data_filename, period=12 * 24 * 7):
    data = np.load(data_filename)['data'][:, :, 0]  # 26208 * 358 * 1
    num_samples, num_nodes = data.shape
    num_train = int(num_samples * 0.6)
    num_ave = int(num_train / period) * period

    time_volume_mx = np.zeros((num_nodes, 7, 288), dtype=np.float32)
    for node in range(num_nodes):
        for i in range(7):
            for t in range(288):
                time_volume = []
                for j in range(i * 288 + t, num_ave, period):
                    time_volume.append(data[j][node])

                time_volume_mx[node][i][t] = np.array(time_volume).mean()

    time_volume_mx = time_volume_mx.reshape(num_nodes, -1)  # (num_nodes, 7*288)
    similarity_mx = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    similarity_mx[:] = np.inf
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            similarity_mx[i][j] = similarity_mx[j][i] = np.sqrt(np.sum((time_volume_mx[i] - time_volume_mx[j]) ** 2))

    distances = similarity_mx[~np.isinf(similarity_mx)].flatten()
    std = distances.std()
    similarity_mx = np.exp(-np.square(similarity_mx / std))
    return [time_volume_mx], similarity_mx


def construct_T(sim_mx, threshold, direct):
    num_nodes = sim_mx.shape[0]
    temporal_graph = np.zeros((num_nodes, num_nodes), dtype=bool)
    for row in range(num_nodes):
        indices = np.argsort(sim_mx[row])[::-1][:threshold]
        temporal_graph[row, indices] = True

    if not direct:
        temporal_graph = np.maximum.reduce([temporal_graph, temporal_graph.T])
        print('构造的时间相似性矩阵是对称的')

    return temporal_graph


def consrtuct_edgelist(distance_df, sensor_ids, filename, weighted=False):
    G = nx.Graph()
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    if weighted:
        for row in distance_df.values:
            if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind or len(row) != 3:
                continue
            G.add_edge(sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]], weight=row[2])
        nx.write_weighted_edgelist(G, filename)
    else:
        for row in distance_df.values:
            if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind or len(row) != 3:
                continue
            G.add_edge(sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]])
        nx.write_edgelist(G, filename, data=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor_ids_filename', type=str, default='',
                        help='File containing sensor ids separated by comma.')
    parser.add_argument('--num_of_vertices', type=int, default=170)  # 03:358, 04:307, 07:883, 08:170
    parser.add_argument('--distances_filename', type=str, default='../data/PEMS08/PEMS08.csv',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--data_filename', type=str, default='../data/PEMS08/PEMS08.npz',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--edgelist_filename', type=str, default='../data/PEMS08/PEMS08.edgelist',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--flow_mean', type=str, default='../data/PEMS08/PEMS08_flow_count.pkl',
                        help='store average flow.')
    parser.add_argument('--thresh_T', type=float, default=15,
                        help='Threshold used in constructing temporal graph.')
    parser.add_argument('--direct_T', type=bool, default=False,
                        help='Whether is the temporal graph directed or undirected.')
    args = parser.parse_args()

    if args.sensor_ids_filename != '':
        with open(args.sensor_ids_filename) as f:
            sensor_ids = f.read().strip().split('\n')
    else:
        sensor_ids = [str(i) for i in range(args.num_of_vertices)]

    distance_df = pd.read_csv(args.distances_filename, dtype={'from': 'str', 'to': 'str'})
    print('Constructing spatial matrix......')
    _, sensor_id_to_ind, adj_mx = get_weighted_adjacency_matrix(distance_df, sensor_ids)

    if not os.path.exists(args.edgelist_filename):
        print('Constructing temporal matrix......')
        time_volume_mx, sim_mx = get_time_volume_matrix(args.data_filename)  # 构造时间相似性矩阵


        print(args.flow_mean)
        with open(args.flow_mean, 'wb') as f:
            pickle.dump(time_volume_mx, f, protocol=2)

        print('Constructing temporal graph......')
        T = construct_T(sim_mx, threshold=args.thresh_T, direct=args.direct_T)  # 变成稀疏矩阵，生成时间图

        print('Constructing spatial graph......')
        consrtuct_edgelist(distance_df, sensor_ids, filename=args.edgelist_filename)  # 根据路网构造空间图

