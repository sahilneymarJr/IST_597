import sys
import tensorflow as tf
import numpy as np
import time
from sklearn.preprocessing import LabelBinarizer
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MLP:

    def __init__(self, size_of_input, hidden_layer_1, hidden_layer_2, output_layer, seed, device):

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
        self.W1 = tf.Variable(tf.random.normal([self.input_size, self.input_size]))
        # Initialize biases for the input layer
        self.b1 = tf.Variable(tf.random.normal([1, self.input_size]))
        # Initialize weights between input layer and hidden layer
        self.W2 = tf.Variable(tf.random.normal([self.input_size, self.hidden_layers_1]))
        # Initialize biases for hidden layer
        self.b2 = tf.Variable(tf.random.normal([1, self.hidden_layers_1]))
        # Initialize weights between hidden layer and second hidden layer
        self.W3 = tf.Variable(tf.random.normal([self.hidden_layers_1, self.hidden_layers_2]))
        # Initialize biases for second hidden layer
        self.b3 = tf.Variable(tf.random.normal([1, self.hidden_layers_2]))
        # Initialize weights between second hidden layer and output layer
        self.W4 = tf.Variable(tf.random.normal([self.hidden_layers_2, self.output_layer]))
        # Intialize biases for the output layer
        self.b4 = tf.Variable(tf.random.normal([1, self.output_layer]))
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

        # backward pass
        # optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)
        with tf.GradientTape() as tape:
            predicted = self.forward(X)
            current_loss = self.loss(Y, predicted) + lamda_L1 * self.L2() + lamda_L2 * self.L1()

        grads = tape.gradient(current_loss, self.variables)

        return grads

    def vanillasgd(self, grads, lr):

        # gradient descent, updating weights
        optimizer = tf.keras.optimizers.SGD(lr)
        optimizer.apply_gradients(zip(grads, self.variables))
        # for i in range(0, len(self.variables)):
        #     W = self.variables[i] - lr * grads[i]
        #     self.variables[i].assign(W)

    def L1(self):

        return (tf.abs(tf.math.reduce_sum(self.W1)) + tf.abs(tf.math.reduce_sum(
            self.W2)) + tf.abs(tf.math.reduce_sum(self.W3))
                + tf.abs(tf.math.reduce_sum(self.W4))) / 4

    def L2(self):

        return (tf.math.reduce_sum(tf.math.square(self.W1)) + tf.math.reduce_sum(
            tf.math.square(self.W2)) + tf.math.reduce_sum(tf.math.square(self.W3))
                + tf.math.reduce_sum(tf.math.square(self.W4))) / 4

    def test(self, test_ds, X_test, obj, lamda_l1=0.0, lamda_l2=0.0, epoch=None):


        test_loss_total = tf.Variable(0, dtype=tf.float32)
        acc = 0
        self.lamda_l1 = lamda_l1
        self.lamda_l2 = lamda_l2
        for inputs, outputs in test_ds:
            preds = obj.forward(inputs)
            test_loss_total = test_loss_total + obj.loss(outputs, preds) + self.lamda_l2 * obj.L2() + self.lamda_l1 * obj.L1()
            preds = np.argmax(preds, axis=1)
            outputs = np.argmax(outputs, axis=1)
            acc = acc + len(np.where(np.equal(preds, outputs))[0])
        acc_ = acc / X_test.shape[0]
        if epoch:
            print("Test Accuracy after epoch: " + format(epoch) + " is " + format(acc_))
        else:
            print("Test Accuracy is " + format(acc_))
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

    def final_training(self, lamda_l1, lamda_l2, lr, train_val_ds, test_ds, X_train_val,X_test):

        self.lamda_l1 = lamda_l1
        self.lamda_l2 = lamda_l2
        self.lr = lr
        self.train_ds = train_val_ds
        self.test_ds = test_ds
        accuracy = []
        final_test_loss = []
        final_train_loss = []
        # running same model 10 times with different seeds
        for i in range(1, 11):

            seed = 5364 + i * 100  # changing seeds
            mlp_ = MLP(size_of_input=X_train_val.shape[1] * X_train_val.shape[2], hidden_layer_1=256, hidden_layer_2=128,
                      output_layer=10, seed=seed, device='gpu')

            epochs = 12

            # time_start = time.time()
            train_plot_X = []
            train_plot_Y = []
            test_plot_X = []
            test_plot_Y = []
            print("Starting Model " + format(i))
            for epoch in range(1, epochs + 1):

                total_loss = tf.Variable(0, dtype=tf.float32)

                for input, output in self.train_ds:
                    preds = mlp_.forward(input)  # forward pass
                    total_loss = total_loss + mlp_.loss(output, preds) + self.lamda_l2 * mlp_.L2() + self.lamda_l1 * mlp_.L1()
                    grads = mlp_.backward(input, output, lamda_L1=self.lamda_l1, lamda_L2=self.lamda_l2)
                    mlp_.vanillasgd(grads, self.lr)

                train_loss = np.sum(total_loss) / X_train_val.shape[0]
                train_plot_X.append(epoch)
                train_plot_Y.append(train_loss)
                if epoch == epochs:
                    final_train_loss.append(train_loss)

                print('Loss for epoch = {} - Average CROSS_ENT_LOSS:= {}'.format(epoch, train_loss))

                if epoch % 2 == 0:
                    print("Model will be tested in this epoch")
                    test_loss, test_acc = mlp_.test(self.test_ds, X_test, mlp_, self.lamda_l1, self.lamda_l2, epoch)
                    test_plot_X.append(epoch)
                    test_plot_Y.append(test_loss)
                    print("Model testing complete")
                    if epoch == epochs:
                        accuracy.append(test_acc)
                        final_test_loss.append(test_loss)

            # time_taken = time.time() - time_start
            #
            # print("\nTotal time taken for model " + format(i) + " is " + format(time_taken) + " seconds")
            mlp.save_graph(train_plot_X, train_plot_Y, test_plot_X, test_plot_Y, i)
            print("Model " + format(i) + " completed")
            print("\n\n")

        print(accuracy)
        print(final_train_loss)
        print(final_test_loss)
        print("Variance w.r.t accuracy: " + format(np.var(accuracy)))
        print("Standard error w.r.t accuracy " + format(np.sum(np.square(accuracy - np.mean(accuracy)))))
        print("Variance w.r.t test loss: " + format(np.var(final_test_loss)))
        print("Standard error w.r.t test loss " + format(np.sum(np.square(final_test_loss - np.mean(final_test_loss)))))
        print("Variance w.r.t training loss: " + format(np.var(final_train_loss)))
        print("Standard error w.r.t training loss " + format(np.sum(np.square(final_train_loss - np.mean(final_train_loss)))))
        print("Done")


if __name__ == "__main__":

    mnist = tf.keras.datasets.mnist

    (X_train_val, Y_train_val), (X_test, Y_test) = mnist.load_data()

    # f_mnist = tf.keras.datasets.fashion_mnist
    #
    # (X_train_val, Y_train_val), (X_test, Y_test) = f_mnist.load_data()


    # We further splitted the data in training and validation dataset #48000:12000:10000
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, train_size=0.8)

    #
    # print(X_train.shape), print(Y_train.shape)
    # print(X_val.shape), print(Y_val.shape)
    # print(X_test.shape), print(Y_test.shape)

    # normalizing
    X_train_val = X_train_val.astype(float) / 255.
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.
    X_val = X_val.astype(float) / 255.

    # one hot encoding labels
    lb = LabelBinarizer()
    Y_train_val = lb.fit_transform(Y_train_val)
    Y_train = lb.fit_transform(Y_train)
    Y_test = lb.fit_transform(Y_test)
    Y_val = lb.fit_transform(Y_val)

    # flattening.
    X_train_val_flatten = X_train_val.reshape(X_train_val.shape[0], -1)
    X_train_flatten = X_train.reshape(X_train.shape[0], -1)
    X_val_flatten = X_val.reshape(X_val.shape[0], -1)
    X_test_flatten = X_test.reshape(X_test.shape[0], -1)

    # batching
    train_val_ds = tf.data.Dataset.from_tensor_slices((X_train_val_flatten, Y_train_val)).batch(32)
    train_ds = tf.data.Dataset.from_tensor_slices((X_train_flatten, Y_train)).batch(32)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val_flatten, Y_val)).batch(8)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test_flatten, Y_test)).batch(8)

    accuracy = []
    final_test_loss = []
    final_train_loss = []

    # hyper_params = {'lambda_L1': [0], 'lambda_L2': [1, 100, 10, 0.01, 0.001, 0.0001, 0], 'lr': [1e-4, 1e-3, 1e-2]}
    hyper_params = { 'lambda_L1': [1, 100, 0.1, 10, 0.01, 0], 'lambda_L2': [0],'lr': [1e-4, 1e-3, 1e-2]}
    # hyper_params = {'lambda_L1': [1], 'lambda_L2': [1], 'lr': [1e-4]}


    epochs = 10

    max = -10
    hp = {}
    flag = True
    for lamda_l1 in hyper_params['lambda_L1']:

        for lamda_l2 in hyper_params['lambda_L2']:

            for lr in hyper_params['lr']:
                # time_start = time.time()
                mlp = MLP(size_of_input=X_train.shape[1] * X_train.shape[2], hidden_layer_1=256, hidden_layer_2=128,
                          output_layer=10, seed=5364, device='gpu')
                print("lambda _l1: " + format(lamda_l1) + " " + ", lambda _l2: " + format(
                    lamda_l2) + " " + ", lr: " + format(lr))
                for epoch in range(1, epochs + 1):


                    total_loss = tf.Variable(0, dtype=tf.float32)

                    for input, output in train_ds:
                        preds = mlp.forward(input)  # forward pass
                        total_loss = total_loss + mlp.loss(output, preds) + lamda_l2 * mlp.L2() + lamda_l1 * mlp.L1()
                        grads = mlp.backward(input, output, lamda_L1=lamda_l1, lamda_L2=lamda_l2)
                        mlp.vanillasgd(grads, lr)

                    train_loss = np.sum(total_loss) / X_train.shape[0]

                    print('Loss for epoch = {} - Average CROSS_ENT_LOSS:= {}'.format(epoch, train_loss))


                # time_taken = time.time() - time_start

                # print("\nTotal time taken for model is " + format(time_taken) + " seconds")
                # print("\n\n")

                test_loss, accuracy = mlp.test(val_ds, X_val, mlp, lamda_l1, lamda_l2)

                if accuracy > 0.95:
                    print("Early stopping as accuracy criteria met")
                    hp['lambda_L1'] = lamda_l1
                    hp['Lambda_L2'] = lamda_l2
                    hp['lr'] = lr

                    print("We are choosing lambda _l1: " + format(lamda_l1) + " " + ", lambda _l2: " + format(lamda_l2) + " " + ", lr: " + format(lr))
                    print("Accuracy on validation set ", accuracy)
                    print("Test loss on validation set ", test_loss)

                    print("MOVING TO FINAL PHASE with Tuned Hyper Parameters by early stopping")
                    print("Running the same model 10 times with different seeds")
                    mlp.final_training(lamda_l1, lamda_l2, lr, train_val_ds, test_ds, X_train_val, X_test) #doing final training with tuned hyperparameters and train/test split
                    flag = False
                    print("DONE")
                    exit() #no need to test further as we have found the hyperparameters and exit from all loops

                if accuracy > max:
                    max = accuracy
                    hp['lambda_L1'] = lamda_l1
                    hp['Lambda_L2'] = lamda_l2
                    hp['lr'] = lr

    if flag:
        print("MOVING TO FINAL PHASE with Tuned Hyper Parameters after choosing the best")
        print("We are choosing lambda _l1: " + format(hp['lambda_L1']) + " " + ", lambda _l2: " + format(
            hp['Lambda_L2']) + " " + ", lr: " + format(hp['lr']))
        print("Running the same model 10 times with different seeds")
        mlp.final_training(hp['lambda_L1'], hp['Lambda_L2'], hp['lr'], train_val_ds, test_ds, X_train_val, X_test)
        print("DONE")