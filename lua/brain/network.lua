local matrix = require("brain.matrix")

local M = {}

---@class brain.Network
---@field num_layers integer
---@field biases brain.Matrix[]
---@field weights brain.Matrix[]
M.Network = {}

M.Network.__index = M.Network

--- Randomly shuffle a list in-place using the Fisher-Yates algorithm.
--- This modifies the original list
---@param list any[]
local function shuffle(list)
	for i = #list, 2, -1 do
        local j = math.random(1, i)
        list[i], list[j] = list[j], list[i]
    end
end

---@generic T : number|brain.Matrix
---@param z T
---@return T
local function sigmoid(z)
    if type(z) == "number" then
        return 1 / (1 + math.exp(-z))
    else -- is a table
        local result = {}
        for i, v in ipairs(z) do
            result[i] = sigmoid(v)
        end
        return matrix.tomatrix(result)
    end
end

---@generic T : number|brain.Matrix
---@param z T
---@return T
local function sigmoid_prime(z)
	return sigmoid(z) * (1 - sigmoid(z))
end

--- The list ``sizes`` contains the number of neurons in the
--- respective layers of the network.  For example, if the list
--- was [2, 3, 1] then it would be a three-layer network, with the
--- first layer containing 2 neurons, the second layer 3 neurons,
--- and the third layer 1 neuron.  The biases and weights for the
--- network are initialized randomly, using a Gaussian
--- distribution with mean 0, and variance 1.  Note that the first
--- layer is assumed to be an input layer, and by convention we
--- won't set any biases for those neurons, since biases are only
--- ever used in computing the outputs from later layers.
---@param sizes integer[]
---@return brain.Network
M.Network.new = function(sizes)
    local self = setmetatable({}, M.Network)
    self.num_layers = #sizes
    self.biases = {}
    for i = 2, self.num_layers do
        self.biases[i - 1] = matrix.randn(sizes[i], 1)
    end
    self.weights = {}
    for i = 1, self.num_layers - 1 do
        self.weights[i] = matrix.randn(sizes[i + 1], sizes[i])
    end
	return self
end

---@param a brain.Matrix
function M.Network:feedforward(a)
    for i = 1, self.num_layers - 1 do
        local w = self.weights[i]
        local b = self.biases[i]
        a = sigmoid(w:dot(a) + b)
    end
    return a
end

function M.Network:cost_derivative(output_activations, y)
    return output_activations - y
end

function M.Network:backprop(x, y)
    local nabla_b = {}
    local nabla_w = {}
    for i, b in ipairs(self.biases) do
        nabla_b[i] = matrix.zeros(b.shape)
    end
    for i, w in ipairs(self.weights) do
        nabla_w[i] = matrix.zeros(w.shape)
    end
    -- feedforward
    local activation = x
    local activations = { x } -- list to store all the activations, layer by layer
    local zs = {} -- list to store all the z vectors, layer by layer
    for i = 1, self.num_layers - 1 do
        local w = self.weights[i]
        local b = self.biases[i]
        local z = w:dot(activation) + b
        zs[i] = z
        activation = sigmoid(z)
		activations[i + 1] = activation
    end
    -- backward pass
    local delta = self:cost_derivative(activations[#activations], y) * sigmoid_prime(zs[#zs])
    nabla_b[#nabla_b] = delta
    nabla_w[#nabla_w] = delta:dot(activations[#activations - 1]:transpose())
    for l = 2, self.num_layers - 1 do
        local z = zs[#zs - l + 1]
        local sp = sigmoid_prime(z)
        delta = (self.weights[#self.weights - l + 2]:transpose():dot(delta)) * sp
        nabla_b[#nabla_b - l + 1] = delta
        nabla_w[#nabla_w - l + 1] = delta:dot(activations[#activations - l]:transpose())
    end
    return nabla_b, nabla_w
end

---Update the network's weights and biases by applying
-- gradient descent using backpropagation to a single mini batch.
-- The "mini_batch" is a list of tuples "(x, y)", and "eta"
-- is the learning rate.
---@param mini_batch table[]
---@param eta number
function M.Network:update_mini_batch(mini_batch, eta)
    local nabla_b = {}
    local nabla_w = {}
    for i, b in ipairs(self.biases) do
        nabla_b[i] = matrix.zeros(b.shape)
    end
    for i, w in ipairs(self.weights) do
        nabla_w[i] = matrix.zeros(w.shape)
    end
    for _, v in ipairs(mini_batch) do
        local x, y = unpack(v)
        local delta_nabla_b, delta_nabla_w = self:backprop(x, y)
        for i = 1, #nabla_b do
            nabla_b[i] = nabla_b[i] + delta_nabla_b[i]
        end
        for i = 1, #nabla_w do
            nabla_w[i] = nabla_w[i] + delta_nabla_w[i]
        end
    end
    for i, w in ipairs(self.weights) do
        self.weights[i] = w - (eta / #mini_batch) * nabla_w[i]
    end
    for i, b in ipairs(self.biases) do
        self.biases[i] = b - (eta / #mini_batch) * nabla_b[i]
    end
end

function M.Network:evaluate(test_data)
    local test_results = {}
    for _, v in ipairs(test_data) do
        local x, y = unpack(v)
        local output = self:feedforward(x)
        local max_index = 1
        for i = 2, #output do
            if output[i][1] > output[max_index][1] then
                max_index = i
            end
        end
        table.insert(test_results, { max_index, y[1][1] })
    end
    local correct = 0
    for _, v in ipairs(test_results) do
        if v[1] - 1 == v[2] then correct = correct + 1 end
    end
    return correct
end

---@param training_data table[]
---@param epochs integer number of epochs to train for
---@param mini_batch_size integer size of mini-batches to use when sampling
---@param eta number learning rate
---@param test_data? table[] optional test data to evaluate the network after each epoch
function M.Network:SGD(training_data, epochs, mini_batch_size, eta, test_data)
    local n_test
    local n = #training_data
    if test_data then n_test = #test_data end
    for j = 1, epochs do
        shuffle(training_data)
        local mini_batches = {}
        for k = 1, n, mini_batch_size do
            table.insert(mini_batches, vim.list_slice(training_data, k, math.min(k + mini_batch_size - 1, n)))
        end
        for _, mini_batch in ipairs(mini_batches) do
            self:update_mini_batch(mini_batch, eta)
        end
        if test_data then
            print(string.format("Epoch %s: %s / %s", j, self:evaluate(test_data), n_test))
        else
            print(string.format("Epoch %d complete", j))
        end
    end
end

function M.Network:save(filename)
    local data = {
        num_layers = self.num_layers,
        biases = self.biases,
        weights = self.weights,
    }
    local file = io.open(filename, "w")
	if not file then
		error("Could not open file for writing: " .. filename)
	end
    file:write(vim.mpack.encode(data))
    file:close()
end

function M.Network.load(filename)
    local file = io.open(filename, "r")
	if not file then
		error("Could not open file for reading: " .. filename)
	end
    local data = vim.mpack.decode(file:read("*a"))
    file:close()
	local net = setmetatable({}, M.Network)
    net.num_layers = data.num_layers
    net.biases = data.biases
    net.weights = data.weights
    return net
end

return M
