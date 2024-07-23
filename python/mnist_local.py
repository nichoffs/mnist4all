import numpy as np
from tensor import Tensor, SGD
import gzip
import os
from tqdm import trange


def fetch_mnist():
    def parse_gz_file(filename):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "..", "data", filename)
        with gzip.open(file_path, "rb") as gz:
            return np.frombuffer(gz.read(), dtype=np.uint8).copy()

    X_train = (
        parse_gz_file("train-images-idx3-ubyte.gz")[0x10:]
        .reshape((-1, 28 * 28))
        .astype(np.float32)
    )
    Y_train = parse_gz_file("train-labels-idx1-ubyte.gz")[8:].astype(np.int8)
    X_test = (
        parse_gz_file("t10k-images-idx3-ubyte.gz")[0x10:]
        .reshape((-1, 28 * 28))
        .astype(np.float32)
    )
    Y_test = parse_gz_file("t10k-labels-idx1-ubyte.gz")[8:].astype(np.int8)
    return X_train, Y_train, X_test, Y_test


X_train, Y_train, X_test, Y_test = fetch_mnist()


def layer_init(m, h):
    return (np.random.uniform(-1.0, 1.0, size=(m, h)) / np.sqrt(m * h)).astype(
        np.float32
    )


class MNISTClassifier:
    def __init__(self):
        self.l1 = Tensor(layer_init(784, 128))
        self.l2 = Tensor(layer_init(128, 10))

    def __call__(self, x):
        return x.dot(self.l1).relu().dot(self.l2).logsoftmax()


model = MNISTClassifier()
optim = SGD([model.l1, model.l2], lr=0.01)

BS = 128
losses, accuracies = [], []
for i in (t := trange(1000)):
    samp = np.random.randint(0, X_train.shape[0], size=(BS))

    x = Tensor(X_train[samp])
    Y = Y_train[samp]
    y = np.zeros((len(samp), 10), np.float32)
    y[range(y.shape[0]), Y] = -1.0  # negative for *N*LL loss

    y = Tensor(y)

    # network
    outs = model(x)

    # NLL loss function
    loss = (outs * y).mean()
    loss.backward()
    optim.step()

    cat = np.argmax(outs.buf, axis=1)
    accuracy = (cat == Y).mean()

    # printing
    loss = loss.buf.item()
    losses.append(loss)
    accuracies.append(accuracy)
    t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))


# evaluate
def numpy_eval():
    Y_test_preds_out = model(Tensor(X_test.reshape((-1, 28 * 28))))
    Y_test_preds = np.argmax(Y_test_preds_out.buf, axis=1)
    return (Y_test == Y_test_preds).mean()


accuracy = numpy_eval()
print("test set accuracy is %f" % accuracy)
assert accuracy > 0.95
