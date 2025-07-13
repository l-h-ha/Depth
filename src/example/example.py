import numpy as np
import requests
import io

URL = "https://raw.githubusercontent.com/l-h-ha/MNIST_SET/refs/heads/main/mnist_train_1.csv"

def get_content_bytes(URL: str) -> bytes:
    response = requests.get(url=URL)
    if response.status_code == 200:
        return response.content
    else:
        raise RuntimeError(f"GET request failed, status: {response.text}")

csv_content_bytes = get_content_bytes(URL=URL)
np_array = np.loadtxt(io.BytesIO(csv_content_bytes), delimiter=",", dtype=int)

labels = np_array[:, 0]
pixels = np_array[:, 1:]

num_samples = labels.shape[0]

y_true = np.zeros((num_samples, 10), dtype=np.float32)
y_true[np.arange(num_samples), labels] = 1

batch_size = 20
num_batches = num_samples // batch_size

batched_y_true = y_true.reshape((num_batches, batch_size, 10)).astype(np.float32)
batched_pixels = (pixels.reshape((num_batches, batch_size, 784)).astype(np.float32) / 255) - 0.5

##
##
##

from ..Depth.models import Stack
from ..Depth.layers import Input, AffineMap
from ..Depth.activations import LeakyReLU, Softmax
from ..Depth.losses import FocalLoss
from ..Depth.initializers import He

model = Stack([
    Input(shape=(batch_size, 784)),
    AffineMap(units=32, activation=LeakyReLU(), initializer=He()),
    AffineMap(units=16, activation=LeakyReLU(), initializer=He()),
    AffineMap(units=10, activation=Softmax(stable=True), initializer=He())
])

epochs = 10
model.fit(batched_pixels, batched_y_true, FocalLoss(), epochs, 0.001)