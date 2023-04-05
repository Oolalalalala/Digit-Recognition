import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import load_model


# 載入資料
(x_train, y_train), (x_test, y_test) = mnist.load_data()

(_, IMAGE_WIDTH, IMAGE_HEIGHT) = x_train.shape
IMAGE_CHANNELS = 1

x_test_with_channels = x_test.reshape(x_test.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
x_test_normalized = x_test_with_channels / 255



# 載入模型
model = load_model("digits_recognition.h5")



# 展示
result = model.predict(x_test_normalized)
predictions = np.argmax(result, axis=1)

w, h = 12, 6
fig, axes = plt.subplots(h, w)
fig.set_size_inches(10, 10)
axes = axes.flatten()

for plot_index in range(w * h):
    predicted_label = predictions[plot_index]
    axes[plot_index].set_xticks([])
    axes[plot_index].set_yticks([])
    color_map = 'Greens' if predicted_label == y_test[plot_index] else 'Reds'
    axes[plot_index].imshow(x_test_normalized[plot_index].reshape((IMAGE_WIDTH, IMAGE_HEIGHT)), cmap=color_map)
    axes[plot_index].set_xlabel(predicted_label)

plt.subplots_adjust(hspace=0, wspace=0)
plt.show()
