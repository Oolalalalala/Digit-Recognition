import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.activations import relu, softmax
from keras.initializers.initializers import VarianceScaling
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy



# 處理輸入資料
(x_train, y_train), (x_test, y_test) = mnist.load_data()
(_, IMAGE_WIDTH, IMAGE_HEIGHT) = x_train.shape
IMAGE_CHANNELS = 1

x_train_with_channels = x_train.reshape(x_train.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
x_test_with_channels = x_test.reshape(x_test.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)

x_train_normalized = x_train_with_channels / 255
x_test_normalized = x_test_with_channels / 255



# 建構模型
model = Sequential(name="Digit-Recognition-Model")

model.add(Conv2D(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
                 kernel_size=5, filters=8, strides=1, activation=relu, kernel_initializer=VarianceScaling()))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(kernel_size=5, filters=16, strides=1, activation=relu, kernel_initializer=VarianceScaling()))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())

model.add(Dense(units=128, activation=relu));

model.add(Dropout(0.2))

model.add(Dense(units=10, activation=softmax, kernel_initializer=VarianceScaling()))

model.summary();



# 編譯
model.compile(optimizer=Adam(learning_rate=0.001), loss=sparse_categorical_crossentropy, metrics=['accuracy'])
# 訓練
model.fit(x_train_normalized, y_train, epochs=10, validation_data=(x_test_normalized, y_test))
# 檢驗
validation_loss, validation_accuracy = model.evaluate(x_test_normalized, y_test)
print('Validation loss: ', validation_loss)
print('Validation accuracy: ', validation_accuracy)

# 儲存
ans = input('Save model? Y/N\n')

if ans == 'Y' or ans == 'y':
    model.save('digits_recognition.h5', save_format='h5')
    print('Model saved')
else:
    print('Model not saved')