from flask import Flask, request, render_template, url_for, jsonify
from keras.models import load_model
from PIL import Image
import numpy as np
from keras.utils import load_img, img_to_array


app = Flask(__name__)

# Plant_Diseases Classification Preprocessing
def img_preprossing(image):
    image=Image.open(image)
    image = image.resize((128, 128))
    image_arr = np.array(image.convert('RGB'))
    image_arr.shape = (1, 128, 128, 3)
    return image_arr


plant_model = load_model('CNN_Basic_model.h5')

# ================================================================================

@app.route('/')
def index():

    return render_template('index.html', appName="Plant Diseases Classification")


#  Skin Classification Diseases
@app.route('/predict_plant_Api', methods=["POST"])
def skin_api():
    # Get the image from post request
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        print("image loaded....") 
        image_arr= img_preprossing(image)
        print("predicting ...")
        new_predict = plant_model.predict(image_arr)
        new_predict = np.argmax(new_predict)
    
        classes = {
    
            0: 'Cherry_healthy',
            1: 'Cherry_Disease',
            2: 'Peach_healthy',
            3: 'Peach_Disease',
            4: 'Pepperbell_Disease',
            5: 'Pepperbell_healthy',
            6: 'Strawberry_healthy',
            7: 'Strawberry_Disease'
        }

        print(classes[new_predict])

        # print(classes[Class])
        prediction = classes[new_predict]
        # print("Model predicting ...")
        # result = model.predict(image_arr)
        # print("Model predicted")
        # Class = np.round(result).astype('int32')
        # prediction = classes[Class]
        # print(prediction)
        return jsonify({'prediction': prediction})
    except:
        return jsonify({'Error': 'Error occur'})


@app.route('/predict_plant_disease', methods=['GET', 'POST'])
def skin_predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        img = request.files['fileup']
        print("image loaded....")
        image_arr= img_preprossing(img)
        print("predicting ...")
        new_predict = plant_model.predict(image_arr)
        new_predict = plant_model.predict(image_arr)
        new_predict = np.argmax(new_predict)
    
        classes = {
    
            0: 'Cherry_healthy',
            1: 'Cherry_Disease',
            2: 'Peach_healthy',
            3: 'Peach_Disease',
            4: 'Pepperbell_Disease',
            5: 'Pepperbell_healthy',
            6: 'Strawberry_healthy',
            7: 'Strawberry_Disease'
        }

        print(classes[new_predict])

        prediction = classes[new_predict]

        return render_template('index.html', prediction=prediction, appName="Plant Diseases Classification")
    else:
        return render_template('index.html',appName="Plant Diseases Classification")


if __name__ == '__main__':
    app.run(debug=True)