local matrix = require("brain.matrix")

local M = {}

---@param filename string Path to the .mpk file
---@return table {training_data, validation_data, test_data}
function M.load_data(filename)
    -- Use vim.system():wait() to decompress the gzipped file synchronously.
    local result = vim.system({'gzip', '-dc', filename}):wait()

    if result.code ~= 0 then
        error("Failed to decompress file: " .. filename .. ". Error: " .. result.stderr)
    end
    local content = result.stdout
    if content == "" then
        error("Decompressed content is empty for file: " .. filename)
    end

    local data = vim.mpack.decode(content)

    local function to_matrix_pair(raw_list, is_training)
        local processed = {}
        for _, item in ipairs(raw_list) do
            local x_raw = item[1]
            local y_raw = item[2]

            -- Convert input to 784x1 matrix
            local x = {}
            for i = 1, #x_raw do
                x[i] = { x_raw[i] }
            end
            x = matrix.tomatrix(x)

            local y
            if is_training then
                -- y_raw is already a vectorized result (e.g., {{0.0}, {1.0}, ...})
                y = matrix.tomatrix(y_raw)
            else
                -- y is a scalar integer (the label), but evaluate expects it in a specific format
                -- In network.lua: evaluate(test_data) expects y[1][1] to be the digit
                y = matrix.tomatrix({{y_raw}})
            end

            table.insert(processed, { x, y })
        end
        return processed
    end

    print("Processing training data...")
    local training_data = to_matrix_pair(data.training_data, true)
    print("Processing validation data...")
    local validation_data = to_matrix_pair(data.validation_data, false)
    print("Processing test data...")
    local test_data = to_matrix_pair(data.test_data, false)

    return training_data, validation_data, test_data
end

return M
