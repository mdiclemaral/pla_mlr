import random
import numpy as np
import matplotlib.pyplot as plt
import time
import sys


###########    P A R T  I    ###########


class Point:   # Randomly generated points are stored.

    def __init__(self, data, feature):
        self.feature = feature
        self.data = data


def PLA(inputs):  # PLA algorithm
    w = np.array([0, 0, 0])
    count = 0

    while 1:
        x, y, count = find_misclassified(w, inputs, count)
        if x is None:
            #print(count)
            return w
        w = w + (y * x)


def find_misclassified(w, inputs, count):  # Finds misclassified samples for PLA algorithm.

    for point in inputs:
        y = point.feature
        x = point.data

        if np.sign(np.dot(w.T, x)) != np.sign(y):
            count+=1
            return x, y, count
    return None, None, count


def generate_random(input_num):  # Generates random data points
    inputs = []
    p_x = []
    p_y = []
    n_x = []
    n_y = []
    random.seed(10)
    for i in range(0, input_num):
        x1 = random.uniform(0, 1)
        p_x.append(x1)
        border_p = -3 * x1 + 1
        y1 = random.uniform(-5, border_p)
        p_y.append(y1)
        positive_point = np.array([1, x1, y1])
        point_p = Point(positive_point, 1)
        inputs.append(point_p)

        x2 = random.uniform(0, 1)
        n_x.append(x2)
        border_n = -3 * x2 + 1
        y2 = random.uniform(border_n, 5)
        n_y.append(y2)
        negative_point = np.array([1, x2, y2])
        point_n = Point(negative_point, -1)
        inputs.append(point_n)
    return inputs, p_x, p_y, n_x, n_y


def plot(w, p_x, p_y, n_x, n_y, fig_name):  # Plots the PLA curve and randomly generated data points
    slopee = -w[1]/w[2]
    p1 = [0, -w[0]/w[2]]

    fig, ax = plt.subplots()

    line2 = plt.axline((0, 1), slope=-3, color="green")
    line1 = plt.axline((p1[0], p1[1]), slope=slopee, color="purple") #, label="p(x)"

    class2 = ax.scatter(p_x, p_y, s=2, c="red")
    class1 = ax.scatter(n_x, n_y, s=2, c="blue")

    ax.legend([line1, line2, class1, class2], ['p(x)', 'f(x)','class 1','class 2'])

    plt.savefig(fig_name)

###########    P A R T  II    ###########

def file_handler(path):  # Extracts data from the given csv file to the numpy arrays

    data_set = np.genfromtxt(path, delimiter=',')
    Y = data_set[:, -1]
    Y = Y[:, np.newaxis]
    data_size = Y.shape[0]

    X = data_set[:, :-1]
    for_t0 = np.ones((data_size,1))
    X = np.concatenate([X, for_t0], axis=1)
    return X, Y


def predict(X, w):  # Predicts the y value for given x and w* values

    result = np.dot(X,w)     #X.w
    return result


def MLR(X, Y):  # Multiple linear regression algorithm.

    temp_a =np.dot(X.T, X)
    temp_b = np.dot(X.T, Y)
    result = np.dot(np.linalg.inv(temp_a), temp_b)

    return result


def Erms(x, y, reg):  # Calculates E_rms for training and test data

    per = int(y.size * 80/100)

    training_x, test_x = x[:per, :], x[per:, :]
    training_t, test_t = y[:per, :], y[per:, :]

    w = MLR_reg(training_x, training_t, reg)

    test_y = predict(test_x, w)
    training_y = predict(training_x, w)

    temp_train = 0
    for i in range(0, len(test_y)):
        temp_train += np.square(training_y[i] - training_t[i])
    temp_train = temp_train/len(training_y)

    temp_test = 0
    for i in range(0, len(test_y)):
        temp_test += np.square(test_y[i] - test_t[i])
    temp_test = temp_test / len(test_y)

    return temp_train, temp_test


def find_lambda(x, y):  # Finds the best lambda value among different regularization values by calculating the minimum E_rms point

    E_test = []
    E_train = []

    for l in range(0, 800):
        train, test = Erms(x, y, l)
        E_train.append(train)
        E_test.append(test)

    ll = list(range(0, 800))
    line1 = plt.scatter(ll, E_train, s=2, c="red")
    line2 = plt.scatter(ll, E_test, s=2, c="blue")

    plt.legend([line1, line2], ['Training Data', 'Test Data'])
    plt.title("E_RMS")
    plt.ylabel("E_rms")
    plt.xlabel("Lambda")
    plt.savefig("E_rms")


    best_lambda = E_train.index(min(E_train))

    return best_lambda


def MLR_reg(X, Y, l):  # Multiple linear regression with regularization
    feature_size = X.shape[1]

    temp_a = np.dot(X.T, X) + (l * np.identity(feature_size))
    temp_b = np.dot(X.T, Y)
    result = np.dot(np.linalg.inv(temp_a), temp_b)

    return result


part = sys.argv[1]
step = sys.argv[2]

if part == "part1":
    input_count = 0

    if step == "step1":
        input_count = 25
        new_inputs, p_x, p_y, n_x, n_y = generate_random(input_count)
        w_last = PLA(new_inputs)
        plot(w_last, p_x, p_y, n_x, n_y, "part1_step1.png")
    elif step == "step2":
        input_count = 50
        new_inputs, p_x, p_y, n_x, n_y = generate_random(input_count)
        w_last = PLA(new_inputs)
        plot(w_last, p_x, p_y, n_x, n_y, "part1_step2.png")
    elif step == "step3":
        input_count = 2500
        new_inputs, p_x, p_y, n_x, n_y = generate_random(input_count)
        w_last = PLA(new_inputs)
        plot(w_last, p_x, p_y, n_x, n_y, "part1_step3.png")

elif part == "part2":
    if step == "step1":
        file_name = 'ds1.csv'
        X, Y = file_handler(file_name)
        start_time = time.time()
        MLR(X, Y)
        print("Time to complete step1: %s msec" % round(1000*(time.time() - start_time)))
    elif step == "step2":
        file_name = 'ds2.csv'
        X, Y = file_handler(file_name)
        start_time = time.time()
        MLR(X, Y)
        print("Time to complete step2: %s msec" % round(1000 * (time.time() - start_time)))
    else:
        file_name = 'ds2.csv'
        X, Y = file_handler(file_name)
        start_time = time.time()
        #best_l = find_lambda(X, Y)
        MLR_reg(X, Y, 459)
        print("Time to complete step3: %s msec" % round(1000*(time.time() - start_time)))
