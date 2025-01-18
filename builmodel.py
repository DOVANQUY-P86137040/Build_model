import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model

# Yêu cầu 1: Build model --- Sequential
sequential_model = models.Sequential([
    layers.Input(shape=(28, 28, 3)),
    layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
    layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
    layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(5, activation='softmax')
])

# Display summary
print("Sequential Model Summary:")
sequential_model.summary()

# Save the structure to an image
plot_model(sequential_model, to_file="sequential_model_structure.png", show_shapes=True)

# Yêu cầu 2: Build model --- Functional API (1)
input_layer = layers.Input(shape=(28, 28, 3))
x = layers.Conv2D(256, kernel_size=(3, 3), activation='relu')(input_layer)
x = layers.Conv2D(256, kernel_size=(3, 3), activation='relu')(x)
x = layers.Conv2D(256, kernel_size=(3, 3), activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)
output_layer = layers.Dense(5, activation='softmax')(x)
functional_model_1 = models.Model(inputs=input_layer, outputs=output_layer)

# Display summary
print("Functional API Model (1) Summary:")
functional_model_1.summary()

# Save the structure to an image
plot_model(functional_model_1, to_file="functional_model_1_structure.png", show_shapes=True)

# Yêu cầu 3: Build model --- Functional API (2) (Split structure)
input_layer = layers.Input(shape=(28, 28, 3))
x = layers.Conv2D(256, kernel_size=(3, 3), activation='relu')(input_layer)
x1 = layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
x2 = layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
x = layers.concatenate([x1, x2])
x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)
output_layer = layers.Dense(5, activation='softmax')(x)
functional_model_2 = models.Model(inputs=input_layer, outputs=output_layer)

# Display summary
print("Functional API Model (2) Summary:")
functional_model_2.summary()

# Save the structure to an image
plot_model(functional_model_2, to_file="functional_model_2_structure.png", show_shapes=True)

# Yêu cầu 4: Build model --- Functional API (3) (Dual input)
input1 = layers.Input(shape=(2804, 20))
input2 = layers.Input(shape=(1024, 4))
shared_conv = layers.Conv1D(64, kernel_size=3, activation='relu')

x1 = shared_conv(input1)
x2 = shared_conv(input2)

x1 = layers.GlobalMaxPooling1D()(x1)
x2 = layers.GlobalMaxPooling1D()(x2)

x = layers.concatenate([x1, x2])
x = layers.Dense(32, activation='relu')(x)
output_layer = layers.Dense(5, activation='softmax')(x)
functional_model_3 = models.Model(inputs=[input1, input2], outputs=output_layer)

# Display summary
print("Functional API Model (3) Summary:")
functional_model_3.summary()

# Save the structure to an image
plot_model(functional_model_3, to_file="functional_model_3_structure.png", show_shapes=True)

# Yêu cầu 5: Build model --- Functional API (4) (Shared weights, dual input)
input1 = layers.Input(shape=(2804, 20))
input2 = layers.Input(shape=(1024, 4))
shared_conv = layers.Conv1D(64, kernel_size=3, activation='relu')

x1 = shared_conv(input1)
x2 = shared_conv(input2)

x1 = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x1, x1)
x2 = layers.MultiHeadAttention(num_heads=4, key_dim=16)(x2, x2)

x1 = layers.GlobalMaxPooling1D()(x1)
x2 = layers.GlobalMaxPooling1D()(x2)

x = layers.concatenate([x1, x2])
x = layers.Dense(32, activation='relu')(x)
output_layer = layers.Dense(5, activation='softmax')(x)
functional_model_4 = models.Model(inputs=[input1, input2], outputs=output_layer)

# Display summary
print("Functional API Model (4) Summary:")
functional_model_4.summary()

# Save the structure to an image
plot_model(functional_model_4, to_file="functional_model_4_structure.png", show_shapes=True)

#6: Build model --- Model Subclassing
class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = layers.Conv2D(256, kernel_size=(3, 3), activation='relu')
        self.conv2 = layers.Conv2D(256, kernel_size=(3, 3), activation='relu')
        self.conv3 = layers.Conv2D(256, kernel_size=(3, 3), activation='relu')
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(32, activation='relu')
        self.dense2 = layers.Dense(5, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

subclassed_model = CustomModel()
subclassed_model.build(input_shape=(None, 28, 28, 3))

# Display summary
print("Model Subclassing Summary:")
subclassed_model.summary()

# Save the structure to an image
plot_model(subclassed_model, to_file="subclassed_model_structure.png", show_shapes=True)
