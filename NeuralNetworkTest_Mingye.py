# Written by Mingye Fu
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler


class MyNN():
    def __init__(self, BOARD_SIZE):
        # a_hidden, a_out are activation functions
        def build_NN(nb_filter, nb_row, nb_col, layers, learning_rate, input_shape, output_shape, a_hidden, a_out, loss,
                     init_in,
                     init_hidden, init_out):
            model = Sequential()
            model.add(Convolution2D(nb_filter, nb_row, nb_col, input_shape=input_shape, activation=a_hidden,
                                    kernel_initializer=init_in))  # first hidden layer
            model.add(Flatten())
            n_hlayers = len(layers)  # number of hidden layers
            for i in range(1, n_hlayers):
                model.add(Dense(layers[i], activation=a_hidden,
                                kernel_initializer=init_hidden))  # other hidden layers
            model.add(Dense(output_shape, activation=a_out, kernel_initializer=init_out))  # last layer

            # Stochastic gradient descent
            model.compile(loss=loss, optimizer=SGD(learning_rate=learning_rate), metrics=['accuracy'])

            return model

        self.BOARD_SIZE = BOARD_SIZE
        # layers = [n_nodes, n_nodes ...], I just set 3 hidden layers with 8 nodes on each layer
        self.p_model = build_NN(nb_filter=2, nb_row=BOARD_SIZE, nb_col=BOARD_SIZE, layers=[8, 8, 8, 8],
                                learning_rate=10, input_shape=(6, BOARD_SIZE, BOARD_SIZE),
                                output_shape=BOARD_SIZE ** 2 + 1,
                                a_hidden='sigmoid',
                                a_out='sigmoid', loss='mean_squared_error', init_in='random_uniform',
                                init_hidden='random_uniform', init_out='random_uniform')
        self.v_model = build_NN(nb_filter=2, nb_row=BOARD_SIZE, nb_col=BOARD_SIZE, layers=[3, 3], learning_rate=10,
                                input_shape=(6, BOARD_SIZE, BOARD_SIZE), output_shape=1,
                                a_hidden='sigmoid',
                                a_out='sigmoid', loss='mean_squared_error', init_in='random_uniform',
                                init_hidden='random_uniform', init_out='random_uniform')

    def train(self, x_train, p_train, v_train, epochs):
        # self.p_model.fit(x_train, p_train, validation_data=(x_train, p_train), epochs=1,
        #                               batch_size=32)
        # self.v_model.fit(x_train, v_train, validation_data=(x_train, v_train), epochs=1,
        #                  batch_size=32)


        # scale v_train
        v_train_sacled = v_train *0.5+0.5
        # train
        self.p_model.fit(np.array([x_train]), np.array([p_train]), epochs=epochs)
        self.v_model.fit(np.array([x_train]), np.array([[v_train_sacled]]), epochs=epochs)

    def predict(self, x_test):
        self.p_prediction = self.p_model.predict(np.array([x_test]))
        v_prediction_scaled = self.v_model.predict(np.array([x_test]))
        self.v_prediction = np.array([[2*(v_prediction_scaled[0,0]-0.5)]])

    def output(self, state):
        self.predict(state)
        # pVector = 1.0 / (self.BOARD_SIZE ** 2+1) * np.ones(self.BOARD_SIZE ** 2 + 1) # '+1' to include pass
        pVector = self.p_prediction[0]
        v = self.v_prediction[0,0]
        return pVector, v


if __name__ == '__main__':
    myNN = MyNN(4)

    state = np.random.random_sample((6,4,4))
    p_train = np.random.random_sample((4 ** 2 + 1))
    v_train = -0.7



    myNN.train(state, p_train, v_train, epochs = 20)

    pVector, v = myNN.output(state)
    print(pVector, v)