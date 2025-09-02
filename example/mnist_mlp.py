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
pixels = np_array[:, 1:].astype(np.float32)/255.0

num_samples = labels.shape[0]

y_true = np.zeros((num_samples, 10), dtype=np.float32)
y_true[np.arange(num_samples), labels] = 1.
y_true = y_true.astype(np.float32)


##
##
##

from ..Depth2.models import Stack
from ..Depth2.layers import AffineMap
from ..Depth2.activations import LeakyReLU, Softmax
from ..Depth2.losses import FocalLoss
from ..Depth2.initializers import He
from ..Depth2.optimizers import GradientDescent

model = Stack([
    AffineMap(units=32, activation=LeakyReLU(), initializer=He()),
    AffineMap(units=16, activation=LeakyReLU(), initializer=He()),
    AffineMap(units=10, activation=Softmax(stable=True), initializer=He())
])
model.set_device("cpu")

epochs = 10
batch_size = 5000

model.fit(
    X=pixels, 
    Y=y_true, 
    loss=FocalLoss(),
    batch_size=batch_size,
    optimizer=GradientDescent(),
    epochs=epochs,
    learning_rate=1e-6,
    )