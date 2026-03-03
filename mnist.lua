print("PID:", vim.fn.getpid())
vim.opt.rtp:prepend(".")

-- Tune the garbage collector to run more frequently
-- This helps manage memory when creating millions of temporary tables.
collectgarbage("setpause", 150)

local network = require("brain.network")

local matrix = require("brain.matrix")

print("Downloading MNIST dataset...")
vim.cmd([[
!curl -LO https://github.com/Eventual-Inc/mnist-json/raw/master/mnist_handwritten_test.json.gz
!curl -LO https://github.com/Eventual-Inc/mnist-json/raw/master/mnist_handwritten_train.json.gz
]])

local get_training_data = function(file)
    return vim.iter(vim.fn.systemlist("gzip -dc " .. file)):map(function(item)
		item = vim.json.decode(item)
		local image = item.image
		image = vim.iter(image):map(function(pixel) return { pixel / 255 } end):totable()
        local label = item.label
        local label_vector = {}
		for i = 1, 10 do
			label_vector[i] = i == label + 1 and {1} or {0}
		end
		return { matrix.tomatrix(image), matrix.tomatrix(label_vector) }
	end):totable()
end

local get_test_data = function(file)
    return vim.iter(vim.fn.systemlist("gzip -dc " .. file)):map(function(item)
		item = vim.json.decode(item)
		local image = item.image
		image = vim.iter(image):map(function(pixel) return { pixel / 255 } end):totable()
		return { matrix.tomatrix(image), matrix.tomatrix({{ item.label }}) }
	end):totable()
end

print("Loading training and test data...")
local training_data = get_training_data("mnist_handwritten_train.json.gz")
local test_data = get_test_data("mnist_handwritten_test.json.gz")

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
