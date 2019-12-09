import math


class Network:
    def __init__(self, file_path, learning_rate=0.1):
        with open(file_path, 'r') as f:
            meta_data = f.readline().rstrip('\n').split(' ')
            self.hidden_length = int(meta_data[1])
            self.input_length = int(meta_data[0])
            self.output_length = int(meta_data[2])
            self.input_weight = []
            self.output_weight = []
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
        self.alpha = learning_rate

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
        return 1 / (1 + math.exp(-x))

    def save_network(self, path):
        with open(path, 'w') as f:
            f.write(str(self.input_length) + " " + str(self.hidden_length) + " " + str(self.output_length) + "\n")
            for x in self.weight[0]:
                f.write(" ".join([str("{0:.3f}".format(num)) for num in x]))
                f.write("\n")
            for x in self.weight[1]:
                f.write(" ".join([str("{0:.3f}".format(num)) for num in x]))
                f.write("\n")

    def get_output(self):
        return self.node[len(self.node)-1]


def back_prop_learning(examples, network, epoch=1):
    """
    This function implements back-propagates learning of a fully-connected neural network.
    :param examples: Training example pairs.
    :param network: Neural network object.
    :param epoch: Number of epoch that a training will go though.
    :return: Return a trained neural network object.
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
                        weight_sum += network.get_weight(m-1, n, t+1) * network.get_node(m-1, t)
                    weight_sum += (-1) * network.get_weight(m-1, n, 0)
                    actual_val = Network.sigmoid(weight_sum)
                    network.set_node(m, n, actual_val)

            # Backward Propagate Begins
            for j in range(network.get_node_width(output_layer_index)):
                a = network.get_node(output_layer_index, j)
                error[output_layer_index][j] = a * (1 - a) * (example[1][j] - a)
            for j in range(output_layer_index-1, 0, -1):
                for n in range(network.get_node_width(j)):
                    weight_sum = 0.0
                    for t in range(network.get_node_width(j+1)):
                        weight_sum += network.get_weight(j, t, n+1) * error[output_layer_index][t]
                    a = network.get_node(j, n)
                    error[j][n] = a * (1 - a) * weight_sum
            for l in range(output_layer_index):
                for n in range(network.get_node_width(l)):
                    for j in range(network.get_node_width(l+1)):
                        update_weight = network.get_weight(l, j, n+1) + network.alpha * network.get_node(l, n) * error[l+1][j]
                        network.set_weight(l, j, n+1, update_weight)
            for l in range(output_layer_index):
                for n in range(network.get_node_width(l+1)):
                    update_bias = network.get_weight(l, n, 0) + network.alpha * (-1) * error[l+1][n]
                    network.set_weight(l, n, 0, update_bias)
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
    network_file_path = input("Please specify the file path of the initial neural network: ")
    train_file_path = input("Please specify the file path of the training dataset: ")
    test_file_path = input("Please specify the output file path: ")
    epoch_num = int(input("Please specify the number of epoch during training: "))
    rate = float(input("Please specify the learning rate of the network: "))
    train_examples = load_training(train_file_path)
    my_network = Network(network_file_path, rate)
    trained_network = back_prop_learning(train_examples, my_network, epoch=epoch_num)
    trained_network.save_network(test_file_path)
