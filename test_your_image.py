import numpy as np
import cv2
import tensorflow as tf
from keras.preprocessing import image

model = tf.keras.models.load_model('fruits.h5')

img = cv2.imread('input.jpg')
img = cv2.resize(img, (100, 100))
img = img.reshape((100, 100,3))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
result = model.predict(images)
result_classes = result.argmax(axis=-1)

arr =  {'Apple Braeburn': 0, 'Apple Crimson Snow': 1, 'Apple Golden 1': 2, 'Apple Golden 2': 3, 'Apple Golden 3': 4, 'Apple Granny Smith': 5, 'Apple Pink Lady': 6, 'Apple Red 1': 7, 'Apple Red 2': 8, 'Apple Red 3': 9, 'Apple Red Delicious': 10, 'Apple Red Yellow 1': 11, 'Apple Red Yellow 2': 12, 'Apricot': 13, 'Avocado': 14, 'Avocado ripe': 15, 'Banana': 16, 'Banana Lady Finger': 17, 'Banana Red': 18, 'Beetroot': 19, 'Blueberry': 20, 'Cactus fruit': 21, 'Cantaloupe 1': 22, 'Cantaloupe 2': 23, 'Carambula': 24, 'Cauliflower': 25, 'Cherry 1': 26, 'Cherry 2': 27, 'Cherry Rainier': 28, 'Cherry Wax Black': 29, 'Cherry Wax Red': 30, 'Cherry Wax Yellow': 31, 'Chestnut': 32, 'Clementine': 33, 'Cocos': 34, 'Dates': 35, 'Eggplant': 36, 'Ginger Root': 37, 'Granadilla': 38, 'Grape Blue': 39, 'Grape Pink': 40, 'Grape White': 41, 'Grape White 2': 42, 'Grape White 3': 43, 'Grape White 4': 44, 'Grapefruit Pink': 45, 'Grapefruit White': 46, 'Guava': 47, 'Hazelnut': 48, 'Huckleberry': 49, 'Kaki': 50, 'Kiwi': 51, 'Kohlrabi': 52, 'Kumquats': 53, 'Lemon': 54, 'Lemon Meyer': 55, 'Limes': 56, 'Lychee': 57, 'Mandarine': 58, 'Mango': 59, 'Mango Red': 60, 'Mangostan': 61, 'Maracuja': 62, 'Melon Piel de Sapo': 63, 'Mulberry': 64, 'Nectarine': 65, 'Nectarine Flat': 66, 'Nut Forest': 67, 'Nut Pecan': 68, 'Onion Red': 69, 'Onion Red Peeled': 70, 'Onion White': 71, 'Orange': 72, 'Papaya': 73, 'Passion Fruit': 74, 'Peach': 75, 'Peach 2': 76, 'Peach Flat': 77, 'Pear': 78, 'Pear Abate': 79, 'Pear Forelle': 80, 'Pear Kaiser': 81, 'Pear Monster': 82, 'Pear Red': 83, 'Pear Williams': 84, 'Pepino': 85, 'Pepper Green': 86, 'Pepper Red': 87, 'Pepper Yellow': 88, 'Physalis': 89, 'Physalis with Husk': 90, 'Pineapple': 91, 'Pineapple Mini': 92, 'Pitahaya Red': 93, 'Plum': 94, 'Plum 2': 95, 'Plum 3': 96, 'Pomegranate': 97, 'Pomelo Sweetie': 98, 'Potato Red': 99, 'Potato Red Washed': 100, 'Potato Sweet': 101, 'Potato White': 102, 'Quince': 103, 'Rambutan': 104, 'Raspberry': 105, 'Redcurrant': 106, 'Salak': 107, 'Strawberry': 108, 'Strawberry Wedge': 109, 'Tamarillo': 110, 'Tangelo': 111, 'Tomato 1': 112, 'Tomato 2': 113, 'Tomato 3': 114, 'Tomato 4': 115, 'Tomato Cherry Red': 116, 'Tomato Maroon': 117, 'Tomato Yellow': 118, 'Walnut': 119}
arr = list(arr.keys())

file = open("output.txt", 'w')
file.write(arr[result_classes[0]])
file.close()
