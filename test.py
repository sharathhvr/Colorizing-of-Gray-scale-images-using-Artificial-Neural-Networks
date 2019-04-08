from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from matplotlib.pyplot import imsave
from skimage.color import lab2rgb,rgb2lab
import cv2


model=load_model("C:/Users/hvrsh/PycharmProjects/PROJECT_529_NEW/model.h5")

new_array=[]
new_array.append(cv2.resize(img_to_array(load_img('C:/Users/hvrsh/PycharmProjects/TEST COLOR MODEL/5.jpg')),(256, 256)))
new_array = np.array(new_array, dtype=float)
new_array = rgb2lab(1.0 / 255 * new_array)[:, :, :, 0]
new_array = new_array.reshape(new_array.shape + (1,))
color_me=new_array

prediction=model.predict(color_me)
prediction = prediction * 128

# Output colorizations
cur = np.zeros((256, 256, 3))
cur[:,:,0] = color_me[0][:,:,0]
cur[:,:,1:] = prediction[0]
imsave("img_result.png", lab2rgb(cur))
