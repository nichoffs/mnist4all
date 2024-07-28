import numpy as np
from tensor import Tensor, SGD
import urllib.request
import gzip
import asyncio
import websockets
import json


def fetch_mnist_download():
    global X_train, Y_train, X_test, Y_test

    def download_and_parse(url):
        with urllib.request.urlopen(url) as response:
            assert response.status == 200
            with gzip.open(response) as gz:
                return np.frombuffer(gz.read(), dtype=np.uint8).copy()

    BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    X_train = (
        download_and_parse(f"{BASE_URL}train-images-idx3-ubyte.gz")[0x10:]
        .reshape((-1, 28 * 28))
        .astype(np.float32)
    )
    Y_train = download_and_parse(f"{BASE_URL}train-labels-idx1-ubyte.gz")[8:].astype(
        np.int8
    )
    X_test = (
        download_and_parse(f"{BASE_URL}t10k-images-idx3-ubyte.gz")[0x10:]
        .reshape((-1, 28 * 28))
        .astype(np.float32)
    )
    Y_test = download_and_parse(f"{BASE_URL}t10k-labels-idx1-ubyte.gz")[8:].astype(
        np.int8
    )
    return "Dataset downloaded successfully"


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


model = None


def inference(image_data):
    global model
    if model is None:
        return "Model not trained. Please train the model first."

    # Convert the received image data to a Tensor
    data = np.array(image_data, dtype=np.float32).reshape(1, 784)
    image_tensor = Tensor(data)

    # Perform inference
    output = model(image_tensor)

    # Get the predicted digit
    predicted_digit = np.argmax(output.buf)

    return int(predicted_digit)


async def train_model(websocket):
    global X_train, Y_train, X_test, Y_test
    if X_train is None or Y_train is None or X_test is None or Y_test is None:
        await websocket.send(
            json.dumps({"error": "Dataset not downloaded. Please download first."})
        )
        return

    model = MNISTClassifier()
    optim = SGD([model.l1, model.l2], lr=0.01)
    BS = 128

    for i in range(1000):
        samp = np.random.randint(0, X_train.shape[0], size=(BS))
        x = Tensor(X_train[samp])
        Y = Y_train[samp]
        y = np.zeros((len(samp), 10), np.float32)
        y[range(y.shape[0]), Y] = -1.0
        y = Tensor(y)

        outs = model(x)
        loss = (outs * y).mean()
        loss.backward()
        optim.step()

        cat = np.argmax(outs.buf, axis=1)
        accuracy = (cat == Y).mean()

        loss_value = loss.buf.item()

        await websocket.send(
            json.dumps({"iteration": i, "loss": loss_value, "accuracy": accuracy})
        )

    # Final evaluation
    Y_test_preds_out = model(Tensor(X_test.reshape((-1, 28 * 28))))
    Y_test_preds = np.argmax(Y_test_preds_out.buf, axis=1)
    final_accuracy = (Y_test == Y_test_preds).mean()

    await websocket.send(json.dumps({"final_accuracy": final_accuracy}))


async def websocket_handler(websocket, path):
    global model
    try:
        async for message in websocket:
            data = json.loads(message)
            if data["action"] == "download":
                result = fetch_mnist_download()
                await websocket.send(json.dumps({"download_status": result}))
            elif data["action"] == "train":
                model = MNISTClassifier()  # Initialize the model
                await train_model(websocket)
            elif data["action"] == "inference":
                if "image_data" not in data:
                    await websocket.send(
                        json.dumps({"error": "No image data provided"})
                    )
                else:
                    prediction = inference(data["image_data"])
                    await websocket.send(json.dumps({"prediction": prediction}))
    except websockets.exceptions.ConnectionClosedError:
        pass


async def main():
    server = await websockets.serve(websocket_handler, "localhost", 8765)
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
