import msgpack
import gzip
import numpy as np
import pickle

def load_data():
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    # Use encoding='latin1' for Python 3 to load Python 2 pickled data
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def export_mnist_to_gzipped_msgpack(output_file='data/mnist.mpk.gz'):
    tr_d, va_d, te_d = load_data()

    # Process training data
    training_data_processed = []
    for x, y in zip(tr_d[0], tr_d[1]):
        training_data_processed.append((x.tolist(), vectorized_result(y).tolist()))

    # Process validation data
    validation_data_processed = []
    for x, y in zip(va_d[0], va_d[1]):
        validation_data_processed.append((x.tolist(), int(y)))

    # Process test data
    test_data_processed = []
    for x, y in zip(te_d[0], te_d[1]):
        test_data_processed.append((x.tolist(), int(y)))

    data_to_export = {
        "training_data": training_data_processed,
        "validation_data": validation_data_processed,
        "test_data": test_data_processed,
    }

    # Open a gzip file in write mode and pack the data into it
    with gzip.open(output_file, 'wb') as f:
        msgpack.pack(data_to_export, f)
    print(f"MNIST data exported to {output_file}")

if __name__ == "__main__":
    export_mnist_to_gzipped_msgpack('/home/brianhuster/.config/nvim/pack/mine/opt/nvim-brain/data/mnist_lua.mpk.gz')
