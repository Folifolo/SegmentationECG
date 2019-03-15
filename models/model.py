from keras import backend as K
from keras.layers import (
    Dense,
    Conv1D,
    MaxPooling1D,
    UpSampling1D
)
from keras.models import (
    Sequential
)

def make_model(num_leads_signal = 12):
    model = Sequential()

    model.add(Conv1D(32, kernel_size=8,
                     activation=K.elu,
                     input_shape=(None, num_leads_signal), padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(32, kernel_size=8, activation=K.elu, padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(32, kernel_size=8, activation=K.elu, padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(32, kernel_size=8, activation=K.elu, padding='same'))

    model.add(Conv1D(32, kernel_size=8, activation=K.elu, padding='same'))
    model.add(UpSampling1D(2))
    model.add(Conv1D(32, kernel_size=8, activation=K.elu, padding='same'))
    model.add(UpSampling1D(2))
    model.add(Conv1D(32, kernel_size=8, activation=K.elu, padding='same'))
    model.add(UpSampling1D(2))
    model.add(Conv1D(32, kernel_size=8, activation=K.elu, padding='same'))
    model.add(Dense(4, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
if __name__ == "__main__":
    #from keras.utils import plot_model
    model = make_model()
    #plot_model(model, to_file='model.png')