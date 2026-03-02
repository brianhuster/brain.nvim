vim.opt.rtp:prepend(".")

-- Tune the garbage collector to run more frequently
-- This helps manage memory when creating millions of temporary tables.
collectgarbage("setpause", 150)

local loader = require("brain.loader")
local network = require("brain.network")

print("Loading MNIST data...")
local training_data, validation_data, test_data = loader.load_data("data/mnist.mpk.gz")

-- Create a network with 784 input neurons, 30 neurons in the hidden layer,
-- and 10 output neurons.
print("Creating neural network...")
local net = network.Network.new({784, 30, 10})

-- Train the network using stochastic gradient descent.
print("Starting training...")
net:SGD(training_data, 30, 10, 3.0, test_data)

print("Training complete.")

-- You can optionally save the trained network
print("Saving trained network to data/mnist_net.mpk")
net:save("data/mnist_net.mpk")