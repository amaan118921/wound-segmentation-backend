from io import BytesIO
from firebase_admin import credentials, initialize_app, storage
import requests
from PIL import Image
from keras.models import load_model

from models.deeplab import Deeplabv3, relu6, BilinearUpsampling, DepthwiseConv2D

from utils.learning.metrics import dice_coef, precision, recall
from utils.io.data import save_results, load_test_images, DataGen
import numpy as np

# settings
input_dim_x = 224
input_dim_y = 224
color_space = 'rgb'
path = './data/Medetec_foot_ulcer_224/'
outputPath = './data/output/'
weight_file_name = 'test.hdf5'
pred_save_path = 'test/'

# data_gen = DataGen(path, split_ratio=0.0, x=input_dim_x, y=input_dim_y, color_space=color_space)
# x_test, test_label_filenames_list = load_test_images(path)

model = load_model('./training_history/' + weight_file_name
                   , custom_objects={'recall': recall,
                                     'precision': precision,
                                     'dice_coef': dice_coef,
                                     'relu6': relu6,
                                     'DepthwiseConv2D': DepthwiseConv2D,
                                     'BilinearUpsampling': BilinearUpsampling})


# for image_batch, label_batch in data_gen.generate_data(batch_size=len(x_test), test=True):
#     prediction = model.predict(image_batch, verbose=1)
#     print(len(prediction))
#     save_results(prediction, 'rgb', outputPath, test_label_filenames_list)
#     break

# input_image_path = './data/Medetec_foot_ulcer_224/test/images/img1.png'
# input_image = Image.open(input_image_path)
# input_image = input_image.resize((input_dim_x, input_dim_y))
# input_image = np.array(input_image) / 255.0  # Normalize the image (assuming pixel values are in [0, 255])
#
# # Predict using the model
# prediction = model.predict(np.expand_dims(input_image, axis=0))
# test_label_filenames_list = ['res.png']
# # Save the prediction result
# save_results(prediction, 'rgb', outputPath, test_label_filenames_list)

# Init firebase with your credentials


def init():
    cred = credentials.Certificate('./womensafety-c4d41-1573ac3bb347.json')
    initialize_app(cred, {'storageBucket': 'womensafety-c4d41.appspot.com'})


def upload_img(filename, file):
    init()
    bucket = storage.bucket()
    blob = bucket.blob(file)
    blob.upload_from_filename(filename)

    blob.make_public()

    print("your file url", blob.public_url)


def predict_result(image_url, filename):
    response = requests.get(image_url)
    if response.status_code == 200:
        image_data = response.content
        input_image = Image.open(BytesIO(image_data))
        input_image = input_image.resize((input_dim_x, input_dim_y))
        input_image = np.array(input_image) / 255.0  # Normalize the image (assuming pixel values are in [0, 255])

        # Predict using the model
        prediction = model.predict(np.expand_dims(input_image, axis=0))
        test_label_filenames_list = [filename]

        # Save the prediction result
        save_results(prediction, 'rgb', outputPath, test_label_filenames_list)
        try:
            upload_img(outputPath + filename, 'images/' + filename)
            return True
        except Exception as e:
            print("failed")
            return False

    else:
        print("Failed to download the image from the URL: {image_url}")
