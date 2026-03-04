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
    local pipe = io.popen("gzip -dc " .. file)
    if not pipe then
        error("Could not open file for reading: " .. file)
    end
    local items = {}
    local json_line = ""
    local i = 1
    while true do
        json_line = pipe:read("*l")
        if not json_line then break end
        local item = vim.json.decode(json_line)
        local image = item.image
        image = vim.iter(image):map(function(pixel) return { pixel / 255 } end):totable()
        local label = item.label
        local label_vector = {}
        for j = 1, 10 do
            label_vector[j] = j == label + 1 and { 1 } or { 0 }
        end
        items[i] = { matrix.tomatrix(image), matrix.tomatrix(label_vector) }
		i = i + 1
    end
    pipe:close()
    return items
end

local get_test_data = function(file)
    local pipe = io.popen("gzip -dc " .. file)
    if not pipe then
        error("Could not open file for reading: " .. file)
    end
    local items = {}
    local json_line = ""
    local i = 1
    while true do
        json_line = pipe:read("*l")
        if not json_line then break end
        local item = vim.json.decode(json_line)
        local image = item.image
        image = vim.iter(image):map(function(pixel) return { pixel / 255 } end):totable()
        items[i] = { matrix.tomatrix(image), item.label + 1 }
		i = i + 1
    end
    pipe:close()
    return items
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
