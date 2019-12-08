import math


class Network:
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            meta_data = f.readline().split(' ')
            self.hidden_length = int(meta_data[0])
            self.input_length = int(meta_data[1])
            self.output_length = int(meta_data[2])
            self.input_weight = []
            for i in range(self.input_length):
                node_weight = [float(x) for x in f.readline().split(' ')]
                self.input_weight.append(node_weight)
            self.output_weight = f.readline().split(' ')
        self.weight = []
        self.weight.append(self.input_weight)
        self.weight.append(self.output_weight)
        self.node = [[0.0] * self.input_length, [0.0] * self.hidden_length, [0.0] * self.output_length]
        self.node_input = [[0.0] * self.hidden_length, [0.0] * self.output_length]
        self.alpha = 0.1

    def get_weight(self, i, j):
        return self.weight[i][j]

    def set_weight(self, i, j, value):
        self.weight[i][j] = value

    def set_node(self, i, j, value):
        self.node[i][j] = value

    def get_node(self, i, j):
        return self.node[i][j]

    def set_node_input(self, i, j, value):
        self.node_input[i][j] = value

    def get_node_input(self, i, j):
        return self.node_input[i][j]

    def get_node_width(self, i):
        return len(self.node[i])

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(x))

    def save_network(self, path):
        with open(path, 'w') as f:
            for l in range(len(self.weight)):
                f.write(str(self.weight[l]))
                f.write("\n")


def back_prop_learning(examples, network, epoch=1):
    """

    :param examples:
    :param network:
    :param epoch:
    :return:
    """
    L = 3
    output_layer_index = 2
    error = [[0.0] * network.get_node_width(0), [0.0] * network.get_node_width(1), [0.0] * network.get_node_width(output_layer_index)]
    for k in range(epoch):
        for j in range(network.get_node_width(0)):
            network.set_node(0, j, examples[0][j])
        for m in range(start=1, stop=L):
            for n in range(network.get_node_width(m)):
                weight_sum = 0.0
                for t in range(network.get_node_width(m-1)):
                    weight_sum += network.get_weight(m-1, t) * network.get_node(m-1, t)
                network.set_node_input(m, n, weight_sum)
                actual_val = Network.sigmoid(weight_sum)
                network.set_node(m, n, actual_val)
        for j in range(network.get_node_width(output_layer_index)):
            a = network.get_node(output_layer_index, j)
            error[output_layer_index][j] = a * (1 - a) * (examples[1][j] - a)
        for j in range(start=L-1, stop=output_layer_index, step=-1):
            for n in range(network.get_node_width(j)):
                weight_sum = 0.0
                for t in range(network.get_node_width(j+1)):
                    weight_sum += network.get_weight(j, t) * error[output_layer_index][t]
                a = network.get_node(j, n)
                error[j][n] = a * (1 - a) * weight_sum
        for l in range(output_layer_index):
            for n in range(network.get_node_width(l)):
                for j in range(network.get_node(l+1)):
                    update_weight = network.get_weight(l, n) + network.alpha * network.get_node(l, n) * error[l+1][j]
                    network.set_node(update_weight)
    return network
