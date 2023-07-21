import os
import logging
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def get_logger(log_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(stream_handler)

    return logger


def calculate_metrics(predicted_labels, true_labels, task_type=2):
    true_labels_np = true_labels
    predicted_labels_np = predicted_labels

    if task_type > 2:
        # true_labels_np = true_labels_np.argmax(dim=1)
        # predicted_labels_np = predicted_labels_np.argmax(dim=1)

        f1 = f1_score(true_labels_np, predicted_labels_np, average='macro')
        precision = precision_score(true_labels_np, predicted_labels_np, average='macro')
        recall = recall_score(true_labels_np, predicted_labels_np, average='macro')

    elif task_type == 2:
        true_labels_np = true_labels.squeeze()
        predicted_labels_np = torch.argmax(predicted_labels, dim=1).squeeze()

        f1 = f1_score(true_labels_np, predicted_labels_np)
        precision = precision_score(true_labels_np, predicted_labels_np)
        recall = recall_score(true_labels_np, predicted_labels_np)

    else:
        raise ValueError("Invalid task_type. Supported values are 'multi' and 'binary'.")

    return f1, precision, recall


def write_log_file(file_name_path, log_str, print_flag=True):
    if print_flag:
        print(log_str)
    if log_str is None:
        log_str = 'None'
    if os.path.isfile(file_name_path):
        with open(file_name_path, 'a+') as log_file:
            log_file.write(log_str + '\n')
    else:
        with open(file_name_path, 'w+') as log_file:
            log_file.write(log_str + '\n')


def sort_edge_index(edge_index):
    edge_pair_list = [pair for pair in zip(edge_index[0], edge_index[1])]
    edge_pair_list.sort(key=lambda pair: pair[0])
    edge_index = torch.tensor([[pair[0] for pair in edge_pair_list], [pair[1] for pair in edge_pair_list]])
    return edge_index


def get_line_graph(edge_index):
    V_edge_index = edge_index

    V_num_edge = int(len(V_edge_index[0]))
    V_edge_pairs = V_edge_index.T.tolist()

    E_adj = np.zeros((V_num_edge, V_num_edge))
    for i in range(V_num_edge):
        for j in range(i+1, V_num_edge):
            if len(set(V_edge_pairs[i]) & set(V_edge_pairs[j])) == 0:
                E_adj[i, j] = 0
            else:
                E_adj[i, j] = 1
    E_adj = E_adj + E_adj.T

    E_edge_index = np.nonzero(E_adj)
    E_edge_index = torch.tensor(np.array(E_edge_index))
    return E_edge_index
