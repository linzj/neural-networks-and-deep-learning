import network, network2
import mnist_loader

def main1():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

def main2():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network2.Network([784, 100, 10], cost = network2.CrossEntropyCost)
    net.large_weight_initializer()
    net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, lmbda = 5.0, monitor_evaluation_accuracy=True, monitor_training_accuracy=True)
    

if __name__ == '__main__':
    main2()
