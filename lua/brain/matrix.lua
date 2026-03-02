local M = {}

---@class brain.Matrix : { [integer]: integer[] }
---@field shape integer[]
---@field dot fun(self: brain.Matrix, b: brain.Matrix): brain.Matrix

local Matrix = {}

Matrix.__index = Matrix

---@param a integer[][]
---@return brain.Matrix
function M.tomatrix(a)
	if not a.shape then
		a.shape = {}
		local temp = a
		while type(temp) == "table" do
			table.insert(a.shape, #temp)
			temp = temp[1]
		end
	end
    return setmetatable(a, Matrix)
end

--- Generates a matrix of the specified dimensions filled with random numbers
--- drawn from a normal distribution with mean 0 and variance 1.
---@param rows integer
---@param cols integer
---@return integer[][]
function M.randn(rows, cols)
    local array = {}
    for i = 1, rows do
        array[i] = {}
        for j = 1, cols do
            -- Box-Muller transform to generate a random number from a normal distribution
            local u1 = 1 - math.random()
            local u2 = 1 - math.random()
            local z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            array[i][j] = z0
        end
    end
    array.shape = { rows, cols }
    return M.tomatrix(array)
end

---@param shape table
local function zeros_array(shape)
    local array = {}
	if #shape == 1 then
		for j = 1, shape[1] do
			array[j] = 0
		end
	else
		for i = 1, shape[1] do
			array[i] = zeros_array(vim.list_slice(shape, 2))
		end
	end
    return array
end

---@param shape integer[]
function M.zeros(shape)
	return M.tomatrix(zeros_array(shape))
end

---@param a brain.Matrix|number
---@param b brain.Matrix|number
---@return brain.Matrix
function Matrix.__add(a, b)
	local result = {}
    if type(a) == "table" then
        local rows = #a
        local cols = #a[1]
		local type_b = type(b)
        for i = 1, rows do
            result[i] = {}
            for j = 1, cols do
                result[i][j] = a[i][j] + (type_b == "number" and b or b[i][j])
            end
        end
        result.shape = { rows, cols }
        return M.tomatrix(result)
    end
	return b + a
end

---@param a brain.Matrix|number
---@param b brain.Matrix|number
---@return brain.Matrix
function Matrix.__sub(a, b)
    if type(a) == "number" then
        return b * (-1) + a
    end
	local res = M.zeros { #a, #a[1] }
    if type(b) == "number" then
        for i = 1, #a do
            for j = 1, #a[1] do res[i][j] = a[i][j] - b end
        end
        return res
    end
    for i = 1, #a do
        for j = 1, #a[1] do
            res[i][j] = a[i][j] - b[i][j]
  		end
    end
    return M.tomatrix(res)
end

---@param a brain.Matrix|number
---@param b brain.Matrix|number
function Matrix.__mul(a, b)
	if type(a) == "number" then
		return b * a
	end

    local res = M.zeros { #a, #a[1] }
    if type(b) == "number" then
        for i = 1, #a do
            for j = 1, #a[1] do res[i][j] = a[i][j] * b end
        end
        return M.tomatrix(res)
    end
    for i = 1, #a do
        for j = 1, #a[1] do res[i][j] = a[i][j] * b[i][j] end
    end
    return M.tomatrix(res)
end

---@param b brain.Matrix
---@return brain.Matrix
function Matrix:dot(b)
    assert(#self[1] == #b, "Number of columns in the first matrix must equal the number of rows in the second matrix.")
    local result = {}
    local res_rows, res_cols = #self, #b[1] -- Number of rows and cols in the result matrix
    for i = 1, res_rows do
        result[i] = {}
        for j = 1, res_cols do
            local sum = 0
            for k = 1, #b do
                sum = sum + self[i][k] * b[k][j]
            end
            result[i][j] = sum
        end
    end
    result.shape = { res_rows, res_cols }
    return M.tomatrix(result)
end

---@return brain.Matrix
function Matrix:transpose()
	local result = {}
    for i = 1, self.shape[2] do
        result[i] = {}
        for j = 1, self.shape[1] do
            result[i][j] = self[j][i]
        end
    end
    result.shape = { self.shape[2], self.shape[1] }
    return M.tomatrix(result)
end

return M
