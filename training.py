import math


class Network:
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            meta_data = f.readline().rstrip('\n').split(' ')
            self.hidden_length = int(meta_data[1])
            self.input_length = int(meta_data[0])
            self.output_length = int(meta_data[2])
            self.input_weight = []
            self.output_weight = []
            output_weight_temp = []
            for i in range(self.hidden_length):
                node_weight = [float(x) for x in f.readline().rstrip('\n').split(' ')]
                self.input_weight.append(node_weight)
            for i in range(self.output_length):
                output_node_weight = [float(x) for x in f.readline().rstrip('\n').split(' ')]
                self.output_weight.append(output_node_weight)
        self.weight = []
        self.weight.append(self.input_weight)
        self.weight.append(self.output_weight)
        self.node = [[0.0] * self.input_length, [0.0] * self.hidden_length, [0.0] * self.output_length]
        self.alpha = 0.1

    def get_weight(self, l, i, j):
        return self.weight[l][i][j]

    def set_weight(self, l, i, j, value):
        self.weight[l][i][j] = value

    def set_node(self, i, j, value):
        self.node[i][j] = value

    def get_node(self, i, j):
        return self.node[i][j]

    def get_node_width(self, i):
        return len(self.node[i])

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(x))

    def save_network(self, path):
        with open(path, 'w') as f:
            for x in self.weight[0]:
                f.write(str(x))
                f.write("\n")
            for x in self.weight[1]:
                f.write(str(x))
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
        for example in examples:
            for j in range(network.get_node_width(0)):
                network.set_node(0, j, example[0][j])
            for m in range(1, L, 1):
                for n in range(network.get_node_width(m)):
                    weight_sum = 0.0
                    for t in range(network.get_node_width(m-1)):
                        weight_sum += network.get_weight(m-1, n, t) * network.get_node(m-1, t)
                    actual_val = Network.sigmoid(weight_sum)
                    network.set_node(m, n, actual_val)

            # Backward Propagate Begins
            for j in range(network.get_node_width(output_layer_index)):
                a = network.get_node(output_layer_index, j)
                error[output_layer_index][j] = a * (1 - a) * (example[1][j] - a)
            for j in range(L-1, output_layer_index, -1):
                for n in range(network.get_node_width(j)):
                    weight_sum = 0.0
                    for t in range(network.get_node_width(j+1)):
                        weight_sum += network.get_weight(j, t, n) * error[output_layer_index][t]
                    a = network.get_node(j, n)
                    error[j][n] = a * (1 - a) * weight_sum
            for l in range(output_layer_index):
                for n in range(network.get_node_width(l)):
                    for j in range(network.get_node_width(l+1)):
                        update_weight = network.get_weight(l, j, n) + network.alpha * network.get_node(l, n) * error[l+1][j]
                        network.set_weight(l, j, n, update_weight)
    return network


def load_training(file_path):
    examples = []
    with open(file_path, 'r') as f:
        meta_data = f.readline()[:-1].split(" ")
        data_num = int(meta_data[0])
        attrib_num = int(meta_data[1])
        for i in range(data_num):
            data = [float(x) for x in f.readline().rstrip('\n').split(" ")]
            examples.append([data[:attrib_num], data[attrib_num:]])
    return examples


if __name__ == "__main__":
    train_examples = load_training("wdbc.mini_train")
    my_network = Network("sample.NNWDBC.init")
    trained_network = back_prop_learning(train_examples, my_network)
    trained_network.save_network("trained_result")
