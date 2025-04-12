import tensorflow as tf
import json
import cv2
import numpy as np
import moviepy.editor as mp
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import os

class SoccerImageClassifier(tf.keras.Model):
    def __init__(self, num_classes):
        super(SoccerImageClassifier, self).__init__()
        # Load the ResNet model
        resnet_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        resnet_model.trainable = False
        # Create a new layer for the classification task
        classification_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
        # Create a new model
        inputs = resnet_model.input
        outputs = classification_layer(tf.keras.layers.Flatten()(resnet_model.output))
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.summary()
def convert_subclassed_model_to_sequential(subclassed_model):
    sequential_model = tf.keras.Sequential()
    for layer in subclassed_model.layers:
        sequential_model.add(layer)
    sequential_model.set_weights(subclassed_model.get_weights())
    return sequential_model
labels ={0:'Ball out of play',
            1:'Clearance', 
            2:'Corner', 
            3:'Direct free-kick', 
            4:'Foul', 
            5:'Goal', 
            6:"I don't know",
            7:'Indirect free-kick', 
            8:'Kick-off', 
            9:'Offside', 
            10:'Penalty', 
            11:'Red card', 
            12:'Shots off target', 
            13:'Shots on target', 
            14:'Substitution', 
            15:'Throw-in', 
            16:'Yellow card', 
        }
# Process videos from folders
model = SoccerImageClassifier(num_classes=len(labels))
sequential_model=convert_subclassed_model_to_sequential(model)

# Define a dummy input shape (for building the model)
dummy_input = tf.keras.Input(shape=(224, 224, 3))
_ = sequential_model(dummy_input)  # This builds the model
# Display the model summary
#model.summary()

# Prepare training data
# x = np.array([i[0] for i in training_data])
# y = np.array([i[1] for i in training_data])
#print(len(y))
#print(y)

# Compile and train the model
# sequential_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# sequential_model.fit(x, y, epochs=10)

for layer in sequential_model.layers[:100]:  # Example: Unfreeze the first 100 layers
    layer.trainable=True
# Save the trained model
sequential_model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=10,
                                    zoom_range=0.75,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.2,
                                    horizontal_flip=True,
                                    vertical_flip=False,
                                    fill_mode="nearest")

training_set = train_datagen.flow_from_directory('D:/capstone/phase-2/test1',target_size=(224, 224),batch_size=32,class_mode='categorical')
print(training_set)

# fit the model
r = sequential_model.fit_generator(
    training_set,
    epochs=25,
    steps_per_epoch=len(training_set)
)
import tensorflow as tf
from keras.models import load_model
import cv2
from PIL import Image
sequential_model.save('model.h5')


