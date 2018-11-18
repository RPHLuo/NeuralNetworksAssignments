import sklearn as sk
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data

class Hopfield_Network:
    def __init__(self, num_units, total=0, scope='hopfield_network'):
        self.weights = np.zeros([num_units, num_units])
        self.total = total

    def update(self, input, type='hebbian'):
        if type == 'hebbian':
            self.hebbian_update(input)
        elif type=='storkey':
            self.storkey_learning(input)
        return

    def hebbian_update(self, input):
        self.weights += np.outer(input, input) / self.total
        np.fill_diagonal(self.weights, 0)

    def storkey_learning(self, input):
        net = np.dot(self.weights, input)
        pre = np.outer(input, net)
        post = np.outer(net, input)
        self.weights -= np.add(pre, post) / self.total
        return

    def activate(self, input):
        converged = False
        count = 0
        while not converged:
            Oldinput = input
            indexes = list(range(0,len(input)))
            random.shuffle(indexes)
            for i in indexes:
                if input[i] > 0:
                    sum = 0
                    for w in range(0,len(self.weights)):
                        sum += self.weights[w][i]
                    if sum > 0:
                        input[i] = 1
                    else:
                        input[i] = 0
            if np.array_equal(Oldinput, input):
                count += 1
                if count == 3:
                    converged = True
            else:
                count = 0
        return input

def compare_result(result, data, i):
    lowest_error = len(result)
    for index in range(0,i):
        error = 0.
        for r in range(0,len(result)):
            error += abs(data[index][r] - result[r])
        if lowest_error > error:
            lowest_error = error
    return lowest_error / len(result)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

ones = trY[:, 1]
fives = trY[:, 5]
ones_input = [trX[i] for i in range(0,len(trX)) if ones[i] != 0]
fives_input = [trX[i] for i in range(0,len(trX)) if fives[i] != 0]

t_ones = teY[:, 1]
t_fives = teY[:, 5]
t_output_ones_fives = np.vstack((t_ones,t_fives)).T
t_output_ones_fives = [t_output_ones_fives[i] for i in range(0,len(t_output_ones_fives)) if t_fives[i] != 0 or t_ones[i] != 0]
t_input_ones_fives = [teX[i] for i in range(0,len(teX)) if t_ones[i] != 0 or t_fives[i] != 0]

def hopfield_test(training_type='hebbian'):
    training_inputs = [1,2,3,5,8,10]
    for n in training_inputs:
        hopfield = Hopfield_Network(784, n*2)
        culmulative_accuracy = 0.
        random.shuffle(ones_input)
        random.shuffle(fives_input)
        order = list(range(0,len(t_input_ones_fives)))
        for i in range(0,n):
            hopfield.update(ones_input[i], training_type)
            hopfield.update(fives_input[i], training_type)
        for o in order:
            result = hopfield.activate(t_input_ones_fives[o])
            resultSet = fives_input
            if t_output_ones_fives[o][0] == 1:
                resultSet = ones_input
            culmulative_accuracy += compare_result(result, resultSet, n)
        accuracy = 1 - (culmulative_accuracy / len(t_input_ones_fives))
        print(training_type,': inputs: ', n, ' accuracy: ', accuracy)

hopfield_test('storkey')
hopfield_test()
