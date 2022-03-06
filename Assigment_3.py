import sys
import tensorflow as tf
import numpy as np
import time
from sklearn.preprocessing import LabelBinarizer
import os
import matplotlib.pyplot as plt
import math


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class custom_optimizer(object):
    def __init__(self, lr, beta_1=0.9, beta_2=0.999, beta_3=0.999987, epsilon_1=1e-8, epsilon_2=1e-6):

        self.t = 0
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.epsilon_1 = epsilon_1
        self.epsilon_2 = epsilon_2
        self.m = 0.0
        self.v = 0.0
        self.u = 0.0

    def apply_gradients(self, params):

        self.t = self.t + 1
        grads, variables = zip(*params)
        # f = []
        # print(grads[0].numpy().shape)
        # grad_np = np.array(grads[0].numpy().tolist())
        # for grad in grads[1:]:
        #     f.append(grad.numpy())
        grads = np.array(grads, dtype=object)

        self.m = self.beta_1*self.m + (1 - self.beta_1)*grads
        self.v = self.beta_2*self.v + (1 - self.beta_2)*(grads**2)
        self.u = self.beta_3*self.u + (1 - self.beta_3)*(grads**3)
        self.bias_corrected_m = self.m / (1 - self.beta_1 ** self.t)
        self.bias_corrected_v = self.v / (1 - self.beta_2 ** self.t)
        self.bias_corrected_u = self.u / (1 - self.beta_3 ** self.t)

        for i in range(0, len(variables)):
            W = variables[i] - self.lr * self.bias_corrected_m[i] / (tf.sqrt(self.bias_corrected_v[i]) + tf.sign(self.bias_corrected_u[i]*tf.math.pow(tf.abs(self.bias_corrected_u[i]), 1/3)*self.epsilon_1 + self.epsilon_2))
            variables[i].assign(W)

class MLP(object):

    def __init__(self, optimizer, size_of_input, hidden_layer_1, hidden_layer_2, output_layer, seed, device):

        self.optim = optimizer
        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.size_of_input = size_of_input
        self.hidden_layer_1 = hidden_layer_1
        self.hidden_layer_2 = hidden_layer_2
        self.output_layer = output_layer
        self.device = device
        self.initialize_weights(self.size_of_input, self.hidden_layer_1, self.hidden_layer_2, self.output_layer)

    def initialize_weights(self, input, hidden_1, hidden_2, output):

        self.input_size = input
        self.hidden_layers_1 = hidden_1
        self.hidden_layers_2 = hidden_2
        self.output_layer = output


        # Initialize weights between input and the input layer
        # self.W1 = tf.Variable(tf.random.normal([self.input_size, self.input_size]) * 0.05)
        self.W1 = tf.Variable(tf.random.normal([self.input_size, self.input_size], stddev=0.1))
        # Initialize biases for the input layer
        # self.b1 = tf.Variable(tf.random.normal([1, self.input_size]) * 0.05)
        self.b1 = tf.Variable(tf.random.normal([1, self.input_size], stddev=0.1))
        # Initialize weights between input layer and hidden layer
        # self.W2 = tf.Variable(tf.random.normal([self.input_size, self.hidden_layers_1]) * 0.05)
        self.W2 = tf.Variable(tf.random.normal([self.input_size, self.hidden_layers_1], stddev=0.1))
        # Initialize biases for hidden layer
        # self.b2 = tf.Variable(tf.random.normal([1, self.hidden_layers_1]) * 0.05)
        self.b2 = tf.Variable(tf.random.normal([1, self.hidden_layers_1]) * 0.1)
        # Initialize weights between hidden layer and second hidden layer
        # self.W3 = tf.Variable(tf.random.normal([self.hidden_layers_1, self.hidden_layers_2]) * 0.05)
        self.W3 = tf.Variable(tf.random.normal([self.hidden_layers_1, self.hidden_layers_2], stddev=0.1))
        # Initialize biases for second hidden layer

        # self.b3 = tf.Variable(tf.random.normal([1, self.hidden_layers_2]) * 0.05)
        self.b3 = tf.Variable(tf.random.normal([1, self.hidden_layers_2], stddev=0.1))
        # Initialize weights between second hidden layer and output layer
        # self.W4 = tf.Variable(tf.random.normal([self.hidden_layers_2, self.output_layer]) * 0.05)
        self.W4 = tf.Variable(tf.random.normal([self.hidden_layers_2, self.output_layer] ,stddev=0.1))
        # Intialize biases for the output layer
        # self.b4 = tf.Variable(tf.random.normal([1, self.output_layer]) * 0.05)
        self.b4 = tf.Variable(tf.random.normal([1, self.output_layer], stddev=0.1))
        # Define variables to be updated during backpropagation
        self.variables = [self.W1, self.W2, self.W3, self.W4, self.b1, self.b2, self.b3, self.b4]

    def compute_output(self, X):
        # forward propagation
        X_1 = tf.matmul(X, self.W1) + self.b1
        X_2 = tf.nn.relu(X_1)
        X_3 = tf.matmul(X_2, self.W2) + self.b2
        X_4 = tf.nn.relu(X_3)
        X_5 = tf.matmul(X_4, self.W3) + self.b3
        X_6 = tf.nn.relu(X_5)
        X_7 = tf.matmul(X_6, self.W4) + self.b4
        # X = tf.nn.softmax(X)

        return X_7

    def forward(self, X):

        """
         forward pass
         X: Tensor, inputs
         """
        X = tf.cast(X, dtype=tf.float32)
        if self.device is not None:
            with tf.device('gpu:0' if self.device == 'gpu' else 'cpu'):
                self.y = self.compute_output(X)
        else:
            self.y = self.compute_output(X)

        return self.y

    def loss(self, y_true, y_pred):

        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)  # cross entropy loss

        return cce(y_true, y_pred)

    def backward(self, X, Y, lamda_L1=0.0, lamda_L2=0.0):

        with tf.GradientTape() as tape:
            predicted = self.forward(X)
            current_loss = self.loss(Y, predicted) + lamda_L1 * self.L2() + lamda_L2 * self.L1()

        grads = tape.gradient(current_loss, self.variables)

        return grads

    def adam(self, grads):

        self.optim.apply_gradients(zip(grads, self.variables))

    def SGD(self, grads):

        self.optim.apply_gradients(zip(grads, self.variables))

    def rms_prop(self, grads):

        self.optim.apply_gradients(zip(grads, self.variables))

    def custom_optimizer(self, grads):

        self.optim.apply_gradients(zip(grads, self.variables))


    def L1(self):

        return (tf.abs(tf.math.reduce_sum(self.W1)) + tf.abs(tf.math.reduce_sum(
            self.W2)) + tf.abs(tf.math.reduce_sum(self.W3))
                + tf.abs(tf.math.reduce_sum(self.W4))) / 4

    def L2(self):

        return (tf.math.reduce_sum(tf.math.square(self.W1)) + tf.math.reduce_sum(
            tf.math.square(self.W2)) + tf.math.reduce_sum(tf.math.square(self.W3))
                + tf.math.reduce_sum(tf.math.square(self.W4))) / 4

    def test(self, test_ds, epoch, X_test):

        test_loss_total = tf.Variable(0, dtype=tf.float32)
        acc = 0

        for inputs, outputs in test_ds:
            preds = mlp.forward(inputs)
            test_loss_total = test_loss_total + mlp.loss(outputs, preds) + lamda_l2 * mlp.L2() + lamda_l1 * mlp.L1()
            preds = np.argmax(preds, axis=1)
            outputs = np.argmax(outputs, axis=1)
            acc = acc + len(np.where(np.equal(preds, outputs))[0])
        acc_ = acc / X_test.shape[0]
        print("Test Accuracy after epoch: " + format(epoch) + " is " + format(acc_))
        loss = np.sum(test_loss_total.numpy()) / X_test.shape[0]
        print('Test ERROR CEL: {:.4f}'.format(loss))

        return loss, acc_

    def save_graph(self, train_plot_X, train_plot_Y, test_plot_X, test_plot_Y, i):

        plt.plot(train_plot_X, train_plot_Y)
        plt.plot(test_plot_X, test_plot_Y)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(["Training Loss", "Test loss"])
        plt.savefig("Model_losses/model_" + format(i) + ".png")
        plt.close()


if __name__ == "__main__":

    # mnist = tf.keras.datasets.mnist
    #
    # (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    f_mnist = tf.keras.datasets.fashion_mnist

    (X_train, Y_train), (X_test, Y_test) = f_mnist.load_data()

    # normalizing
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.

    # one hot encoding labels
    #
    lb = LabelBinarizer()
    Y_train = lb.fit_transform(Y_train)
    Y_test = lb.transform(Y_test)

    # flattening.

    X_train_flatten = X_train.reshape(X_train.shape[0], -1)
    X_test_flatten = X_test.reshape(X_test.shape[0], -1)

    # batching
    train_ds = tf.data.Dataset.from_tensor_slices((X_train_flatten, Y_train)).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test_flatten, Y_test)).batch(8)

    accuracy = []
    time_avg = []
    final_test_loss = []
    final_train_loss = []
    # running same model 10 times with different seeds
    for i in range(1, 11):


        seed = 5364 + i * 56  # changing seeds
        lr = 1e-2

        # optim = custom_optimizer(lr)

        # optim = tf.keras.optimizers.RMSprop(
        # learning_rate=0.0001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
        #      name='RMSprop')
        #
        # optim = tf.keras.optimizers.SGD(
        #     learning_rate=0.01, momentum=0.0, nesterov=False, name='SGD')
        #
        optim = tf.keras.optimizers.Adam(
            learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')

        mlp = MLP(optim, size_of_input=X_train.shape[1] * X_train.shape[2], hidden_layer_1=256, hidden_layer_2=128,
                  output_layer=10, seed=seed, device='gpu')

        epochs = 12
        lamda_l2 = 0.0
        lamda_l1 = 0.1

        time_start = time.time()
        train_plot_X = []
        train_plot_Y = []
        test_plot_X = []
        test_plot_Y = []
        print("Starting Model " + format(i))
        for epoch in range(1, epochs + 1):

            total_loss = tf.Variable(0, dtype=tf.float32)

            for input, output in train_ds:
                preds = mlp.forward(input)  # forward pass
                total_loss = total_loss + mlp.loss(output, preds) + lamda_l2 * mlp.L2() + lamda_l1 * mlp.L1()
                grads = mlp.backward(input, output, lamda_L1=lamda_l1, lamda_L2=lamda_l2)
                # mlp.custom_optimizer(grads)
                # mlp.rms_prop(grads)
                # mlp.SGD(grads)
                mlp.adam(grads)
            train_loss = np.sum(total_loss) / X_train.shape[0]
            train_plot_X.append(epoch)
            train_plot_Y.append(train_loss)
            if epoch == epochs:
                final_train_loss.append(train_loss)

            print('Loss for epoch = {} - Average CROSS_ENT_LOSS:= {}'.format(epoch, train_loss))

            if epoch % 2 == 0:
                print("Model will be tested in this epoch")
                test_loss, test_acc = mlp.test(test_ds, epoch, X_test)
                test_plot_X.append(epoch)
                test_plot_Y.append(test_loss)
                print("Model testing complete")
                if epoch == epochs:
                    accuracy.append(test_acc)
                    final_test_loss.append(test_loss)

        time_taken = time.time() - time_start
        time_avg.append(time_taken)

        print("\nTotal time taken for model " + format(i) + " is " + format(time_taken) + " seconds")
        mlp.save_graph(train_plot_X, train_plot_Y, test_plot_X, test_plot_Y, i)
        print("Model " + format(i) + " completed")
        print("\n\n")

    print("-------------------------------------------------------------------")
    print("Average Accuracy: " + format(sum(accuracy)/10))
    print("Average time: " + format(sum(time_avg) / 10))
    # print(final_train_loss)
    # print(final_test_loss)
    print("Variance w.r.t accuracy: " + format(np.var(accuracy)))
    print("Standard error w.r.t accuracy " + format(np.sum(np.square(accuracy - np.mean(accuracy)))))
    print("Variance w.r.t test loss: " + format(np.var(final_test_loss)))
    print("Standard error w.r.t test loss " + format(np.sum(np.square(final_test_loss - np.mean(final_test_loss)))))
    print("Variance w.r.t training loss: " + format(np.var(final_train_loss)))
    print("Standard error w.r.t training loss " + format(np.sum(np.square(final_train_loss - np.mean(final_train_loss)))))
    print("Done")
