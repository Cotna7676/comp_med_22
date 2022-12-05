# figure out number of layers in these freaking models (specifically EfficientNet)

import tensorflow as tf

model = tf.keras.applications.EfficientNetB0()
print("B0 ", len(model.layers), len(model.layers) - 4)

model = tf.keras.applications.EfficientNetB3()
print("B3 ", len(model.layers), len(model.layers) - 4)

model = tf.keras.applications.EfficientNetB4()
print("B4 ", len(model.layers), len(model.layers) - 4)

model = tf.keras.applications.EfficientNetB5()
print("B5 ", len(model.layers), len(model.layers) - 4)

model = tf.keras.applications.EfficientNetB7()
print("B7 ", len(model.layers), len(model.layers) - 4)