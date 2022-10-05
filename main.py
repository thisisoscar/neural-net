import numpy as np
import time

np.set_printoptions(suppress=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.w_i_h = np.random.rand(hidden_size, input_size) * 2 - 1
        self.b_h = np.zeros((hidden_size, 1))
        self.w_h_o = np.random.rand(output_size, hidden_size) * 2 - 1
        self.b_o = np.zeros((output_size, 1))
        self.learning_rate = learning_rate

    def feed_forward(self, data, training=False):
        data = np.reshape(data, (len(data), 1))
        # input -> hidden
        hidden = sigmoid(np.dot(self.w_i_h, data) + self.b_h)

        #print(self.w_i_h, '\n', data, self.b_h, '\n', 'ans', hidden, '\n')

        # hidden -> output
        outputs = sigmoid(np.dot(self.w_h_o, hidden) + self.b_o)

        #print(outputs, '\n')

        if not training:
            return outputs
        else:
            return hidden, outputs

    def train(self, data, target):
        data = np.reshape(data, (len(data), 1))
        target = np.reshape(target, (len(target), 1))

        hidden, outputs = self.feed_forward(data, training=True)

        outputs_error = target - outputs
        delta_w_h_o = self.learning_rate * outputs_error * sigmoid_prime(outputs) * np.transpose(hidden)
        self.w_h_o += delta_w_h_o

        delta_b_o = self.learning_rate * outputs_error * sigmoid_prime(outputs)
        self.b_o += delta_b_o

        hidden_error = np.dot(self.w_h_o.T, outputs_error)
        delta_w_i_h = self.learning_rate * hidden_error * sigmoid_prime(hidden) * np.transpose(data)
        self.w_i_h += delta_w_i_h

        delta_b_h = self.learning_rate * hidden_error * sigmoid_prime(hidden)
        self.b_h += delta_b_h

    def unveil(self):
        print('shape:', self.inputs_size, self.hidden_size, self.output_size)
        print('\nweights input -> hidden\n', self.w_i_h)
        print('\nbiases input -> hidden\n', self.b_h)
        print('\nweights hidden -> output\n', self.w_h_o)
        print('\nbiases hidden -> output\n', self.b_o)
        
'''
#EXAMPLE:
nn = NeuralNetwork(2, 1, 2, 0.1)
start = time.perf_counter()
# runs for 5 minutes
while time.perf_counter() - start < 60 * 5:
    training_data = np.random.rand((2))
    if training_data[0] > 0.33 and training_data[1] > 0.6:
        target = [1, 0]
    else:
        target = [0, 1]

    nn.train(training_data, target)

print(nn.feed_forward([0.05, 0.65]))
'''
