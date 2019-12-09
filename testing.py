import math
import training
import copy


def net_testing(examples, network):
    """

    :param examples:
    :param network:
    :param epoch:
    :return:
    """
    L = 3
    evaluation = [0, 0, 0, 0]
    evaluations = [copy.deepcopy(evaluation) for x in range(len(network.get_output()))]
    for i in range(len(examples)):
        for j in range(network.get_node_width(0)):
            network.set_node(0, j, examples[i][0][j])
        for m in range(1, L, 1):
            for n in range(network.get_node_width(m)):
                weight_sum = 0.0
                for t in range(network.get_node_width(m-1)):
                    weight_sum += network.get_weight(m-1, n, t+1) * network.get_node(m-1, t)
                weight_sum += (-1) * network.get_weight(m-1, n, 0)
                actual_val = training.Network.sigmoid(weight_sum)
                network.set_node(m, n, actual_val)
        test_result = network.get_output()
        for k in range(len(test_result)):
            test_num = round(test_result[k])
            target_num = round(examples[i][1][k])
            if test_num == 1 and target_num == 1:
                evaluations[k][0] += 1
            elif test_num == 1 and target_num == 0:
                evaluations[k][1] += 1
            elif test_num == 0 and target_num == 1:
                evaluations[k][2] += 1
            else:
                evaluations[k][3] += 1
    return evaluations


def save_testing(result, file_path):
    micro_data = []
    for i in range(4):
        micro_data.append(sum([x[i] for x in result]))
    result.append(micro_data)
    outputs = []
    for r in result:
        overall_acc = (r[0] + r[3]) / sum(r)
        precision = r[0] / (r[0] + r[1])
        recall = r[0] / (r[0] + r[2])
        f1 = (2 * precision * recall) / (precision + recall)
        outputs.append([r[0], r[1], r[2], r[3], overall_acc, precision, recall, f1])
    macro_data = []
    for i in range(4):
        macro_data.append(sum([x[i+4] for x in outputs[:-1]]) / len(outputs[:-1]))
    result.append(macro_data)
    with open(file_path, "w") as f:
        for row in outputs[:-1]:
            f.write(" ".join([str(x) for x in row[0:4]]))
            f.write(" ")
            f.write(" ".join(["{0:.3f}".format(x) for x in row[4:]]))
            f.write("\n")
        f.write(" ".join(["{0:.3f}".format(x) for x in outputs[-1][4:]]))
        f.write("\n")
        f.write(" ".join(["{0:.3f}".format(x) for x in macro_data]))


def load_testing(file_path):
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
    my_network = training.Network("sample.NNGrades.05.100.trained")
    test_examples = load_testing("grades.test")
    results = net_testing(test_examples, my_network)
    save_testing(results, "tested_result")
