import tensorflow as tf

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

IMAGE_SIZE=[224,224]

train_path=r"C:\Users\rohde\PycharmProjects\pythonProject200000\Plant_Diseases_Dataset(Augmented)\train"
val_path=r"C:\Users\rohde\PycharmProjects\pythonProject200000\Plant_Diseases_Dataset(Augmented)\valid"


# Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG
# Here we will be using imagenet weights

resnet = ResNet152V2(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in resnet.layers:
    layer.trainable = False


  # useful for getting number of output classes
folders = glob(r"C:\Users\rohde\PycharmProjects\pythonProject200000\Plant_Diseases_Dataset(Augmented)\train\**")

# our layers - you can add more if you want
x = Flatten()(resnet.output)

prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=resnet.input, outputs=prediction)

# view the structure of the model
model.summary()


# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory(r"C:\Users\rohde\PycharmProjects\pythonProject200000\Plant_Diseases_Dataset(Augmented)\train",
                                                 target_size = (224, 224),
                                                 batch_size = 128,
                                                 class_mode = 'categorical')


test_set = test_datagen.flow_from_directory(r"C:\Users\rohde\PycharmProjects\pythonProject200000\Plant_Diseases_Dataset(Augmented)\valid",
                                            target_size = (224, 224),
                                            batch_size = 128,
                                            class_mode = 'categorical')


# fit the model
# Run the cell. It will take some time to execute
r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=50,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)



# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal16_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal16_acc')



# save it as a h5 file


from tensorflow.keras.models import load_model

model.save('model_resnet3225.h5')



y_pred = model.predict(test_set)

import numpy as np
y_pred = np.argmax(y_pred, axis=1)



