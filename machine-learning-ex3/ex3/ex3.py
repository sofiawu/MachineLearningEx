import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

input_layer_size = 400
num_labels = 10

def load_data(file_name):
    data = sio.loadmat(file_name)

    return np.array(data['X'], dtype=np.float32) / 255, np.array(data['y'], dtype=np.float32)

def display_data(data):
    m, n = data.shape
    example_width = int(np.floor(np.sqrt(n)))
    example_height = int(np.ceil(n / example_width))

    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig = plt.figure(figsize=(10, 10))
    for i in range(display_rows):
        for j in range(display_cols):
            img_data = data[i * display_cols + j]
            img_data = img_data.reshape(example_height, example_width).T
            ax = fig.add_subplot(display_rows, display_cols, i * display_cols + j + 1)
            ax.imshow(img_data, cmap=plt.cm.gray)
            ax.axis('off')

    fig.tight_layout()

    plt.show()


X, Y = load_data('ex3data1.mat')
print X.shape, Y.shape

select = np.random.choice(X.shape[0], 100, replace=False)
X_select = X[select]

display_data(X_select)
