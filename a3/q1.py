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
            self.storkey_update(input)
        return

    def hebbian_update(self, input):
        self.weights += np.outer(input, input) / self.total
        np.fill_diagonal(self.weights, 0)

    def storkey_update(self, input):
        self.weights += np.outer(input, input) / self.total
        net = np.dot(self.weights, input)
        pre = np.outer(input, net)
        post = pre.T
        self.weights -= np.add(pre, post) / self.total
        np.fill_diagonal(self.weights, 0)

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

def compare_result(result, ones, fives, i):
    lowest_ones_error = np.inf
    lowest_fives_error = np.inf
    for index in range(0,i):
        ones_error = np.linalg.norm(ones[index]-result)
        fives_error = np.linalg.norm(fives[index]-result)
        if lowest_ones_error > ones_error:
            lowest_ones_error = ones_error
        if lowest_fives_error > fives_error:
            lowest_fives_error = fives_error
    if (lowest_ones_error > lowest_fives_error):
        return 5
    elif (lowest_fives_error > lowest_ones_error):
        return 1
    else:
        return 0

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
        order = list(range(0,len(t_input_ones_fives)))
        for i in range(0,n):
            hopfield.update(ones_input[i], training_type)
            hopfield.update(fives_input[i], training_type)
        for o in order:
            result = hopfield.activate(t_input_ones_fives[o])
            answer = 5
            if t_output_ones_fives[o][0] == 1:
                answer = 1
            guess = compare_result(result, ones_input, fives_input, n)
            if guess == answer:
                culmulative_accuracy += 1
        accuracy = culmulative_accuracy / len(order)
        print(training_type,': inputs: ', n, ' accuracy: ', accuracy)


random.shuffle(ones_input)
random.shuffle(fives_input)
hopfield_test()
hopfield_test('storkey')
