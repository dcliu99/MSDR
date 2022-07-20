import argparse
import pickle
import numpy as np
import os


def generate_graph_seq2seq_io_data(
        data, x_offsets, y_offsets
):
    num_samples, num_nodes, _ = data.shape
    data = data[:, :, 0:1]

    x, y = [], []
    x_timeslot = []
    y_timeslot = []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))

    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]

        x.append(x_t)
        y.append(y_t)

        x_timeslot.append((t + x_offsets) % 288)
        y_timeslot.append((t + y_offsets) % 288)

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    y_timeslot = np.stack(y_timeslot, axis=0)
    x_timeslot = np.stack(x_timeslot, axis=0)

    return x, y, x_timeslot, y_timeslot


def generate_train_val_test(args, mean_filename):
    data_seq = np.load(args.traffic_df_filename)['data']

    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y

    x_offsets = np.arange(-(seq_length_x - 1), 1, 1)
    y_offsets = np.arange(args.y_start, (seq_length_y + 1), 1)

    x, y, x_timeslot, y_timeslot = generate_graph_seq2seq_io_data(data=data_seq, x_offsets=x_offsets,
                                                                  y_offsets=y_offsets)

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    print("x_timeslot: ", x_timeslot.shape, ", y_timeslot: ", y_timeslot.shape)
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.6)
    num_val = num_samples - num_train - num_test

    x_train, y_train = x[:num_train], y[:num_train]
    x_timeslot_train = x_timeslot[:num_train]
    y_timeslot_train = x_timeslot[:num_train]

    x_val, y_val = x[num_train:num_train + num_val], y[num_train:num_train + num_val]
    y_timeslot_val = y_timeslot[num_train: num_train + num_val]
    x_timeslot_val = x_timeslot[num_train: num_train + num_val]

    x_test, y_test = x[num_train + num_val:], y[num_train + num_val:]
    y_timeslot_test = y_timeslot[-num_test:]
    x_timeslot_test = x_timeslot[-num_test:]

    with open(mean_filename, 'rb') as f:
        pickle_data = pickle.load(f)
    pd = np.array(pickle_data)
    _, num_nodes, timeslot = pd.shape
    pickle_mean_data = np.reshape(pickle_data, (num_nodes, 7, 288))
    pickle_mean_data = np.mean(pickle_mean_data, axis=1)

    constant = 5

    x_train_len, T, num_nodes, input_dim = x_train.shape
    for index in range(x_train_len):
        x_train_value = x_train[index][T - 1]
        cur_timeslot = x_timeslot_train[index][-1]
        indices = []
        for i in range(num_nodes):
            if x_train_value[i, 0] == 0:
                indices.append(i)

        for ind in indices:
            for t in range(T - 1)[::-1]:
                if x_train[index][t][ind][0] != 0:
                    x_train_value[ind][0] = x_train[index][t][ind][0]
                    break

        for ind in indices:
            if x_train_value[ind, 0] == 0:
                for prev_timeslot in range(cur_timeslot + 1)[::-1]:
                    if pickle_mean_data[ind][prev_timeslot] != 0:
                        x_train_value[ind][0] = pickle_mean_data[ind][cur_timeslot]
                        break

        for ind in indices:
            if x_train_value[ind, 0] == 0:
                x_train_value[ind][0] = constant

        for t in range(T):
            for node in range(num_nodes):
                y_train[index][t][node][0] = (y_train[index][t][node][0] - x_train_value[node][0]) / \
                                             x_train_value[node][0]

    x_val_len, T, num_nodes, input_dim = x_val.shape
    for index in range(x_val_len):
        x_val_value = x_val[index][T - 1]
        cur_timeslot = x_timeslot_val[index][-1]
        indices = []
        for i in range(num_nodes):
            if x_val_value[i, 0] == 0:
                indices.append(i)

        for ind in indices:
            for t in range(T - 1)[::-1]:
                if x_val[index][t][ind][0] != 0:
                    x_val_value[ind][0] = x_val[index][t][ind][0]
                    break

        for ind in indices:
            if x_val_value[ind, 0] == 0:
                for prev_timeslot in range(cur_timeslot + 1)[::-1]:
                    if pickle_mean_data[ind][prev_timeslot] != 0:
                        x_val_value[ind][0] = pickle_mean_data[ind][cur_timeslot]
                        break

        for ind in indices:
            if x_val_value[ind, 0] == 0:
                x_val_value[ind][0] = constant

    x_test_len, T, num_nodes, input_dim = x_test.shape
    for index in range(x_test_len):
        x_test_value = x_test[index][T - 1]
        cur_timeslot = x_timeslot_test[index][-1]
        indices = []
        for i in range(num_nodes):
            if x_test_value[i, 0] == 0:
                indices.append(i)

        for ind in indices:
            for t in range(T - 1)[::-1]:
                if x_test[index][t][ind][0] != 0:
                    x_test_value[ind][0] = x_test[index][t][ind][0]
                    break

        for ind in indices:
            if x_test_value[ind, 0] == 0:
                for prev_timeslot in range(cur_timeslot + 1)[::-1]:
                    if pickle_mean_data[ind][prev_timeslot] != 0:
                        x_test[ind][0] = pickle_mean_data[ind][cur_timeslot]
                        break

        for ind in indices:
            if x_test_value[ind, 0] == 0:
                x_test_value[ind][0] = constant

    for cat in ['train', 'val', 'test']:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        _y_slot = locals()["y_timeslot_" + cat]
        _x_slot = locals()["x_timeslot_" + cat]

        print(cat, "x: ", _x.shape, "y:", _y.shape, "y_slot:", _y_slot.shape)
        np.savez_compressed(

            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_slot=_x_slot,
            y_slot=_y_slot,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="../data/processed/PEMS08/", help="output folder")
    parser.add_argument('--flow_mean', type=str, default="../data/PEMS08/PEMS08_flow_count.pkl", help="mean flow file")
    parser.add_argument('--traffic_df_filename', type=str, default="../data/PEMS08/PEMS08.npz", help="dataset")
    parser.add_argument('--seq_length_x', type=int, default=12, help='input sequence len')
    parser.add_argument('--seq_length_y', type=int, default=12, help='output sequence len')
    parser.add_argument('--y_start', type=int, default=1, help='start step')

    args = parser.parse_args()
    mean_flow_file = args.flow_mean
    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} 存在，是否将其作为输出目录?(y/n)')).lower().strip()
        if reply[0] != 'y':
            exit()
    else:
        os.makedirs(args.output_dir)
    generate_train_val_test(args, mean_flow_file)