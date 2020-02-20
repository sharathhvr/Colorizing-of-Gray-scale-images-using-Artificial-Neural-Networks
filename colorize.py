#Get images
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer,Conv2D,UpSampling2D
from keras.preprocessing.image import img_to_array, load_img
from matplotlib.pyplot import imsave
from skimage.color import rgb2lab,lab2rgb,rgb2gray
import os
import cv2

trainX=[]
trainY=[]
for filename in os.listdir('C:/Users/hvrsh/PycharmProjects/PROJECT_529_NEW/All images/Dog/'):
    image2=img_to_array(load_img('C:/Users/hvrsh/PycharmProjects/PROJECT_529_NEW/All images/Dog/'+filename))
    image=cv2.resize(image2,(256,256))
    image = np.array(image, dtype=float)
    # Import map images into the lab colorspace
    X = rgb2lab(1.0/255*image)[:,:,0]  #L values
    Y = rgb2lab(1.0/255*image)[:,:,1:]  #a,b values
    Y = Y / 128
    trainX.append(X)
    trainY.append(Y)

trainX=np.array(trainX).reshape(-1,256,256,1)
trainY=np.array(trainY).reshape(-1,256,256,2)

# Building the neural network
model = Sequential()
model.add(InputLayer(input_shape=(256, 256, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))

# Finish model
model.compile(optimizer='rmsprop',loss='mse')

#Train the neural network
model.fit(x=trainX, y=trainY, batch_size=1, epochs=300)
print(model.evaluate(trainX, trainY, batch_size=1))

#coloring
color_me = []
for filename in os.listdir('C:/Users/hvrsh/PycharmProjects/PROJECT_529_NEW/All images/Test/'):
    color_me.append(cv2.resize(img_to_array(load_img('C:/Users/hvrsh/PycharmProjects/PROJECT_529_NEW/All images/Test/'+filename)),(256,256)))
    color_me = np.array(color_me, dtype=float)
    color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
    color_me = color_me.reshape(color_me.shape+(1,))

#save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save("model.h5")

# Test model
output = model.predict(color_me)
output = output * 128

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave("img_result1.png", lab2rgb(cur))

