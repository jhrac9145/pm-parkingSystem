from flask import Flask, render_template, jsonify, request
from pymongo import MongoClient
from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

import base64
import numpy as np

app = Flask(__name__)

client = MongoClient('localhost', 27017)
db = client.dbtest

# 불러올 모델 파일의 경로
model_path = 'best_model.h5'

# HTML 화면 보여주기
@app.route('/')
def homework():
    return render_template('index.html')

# 분류결과 호출 기능 추가 코드
# 저장하기(POST) API
@app.route('/api/uploading', methods=['POST'])
def image_upload():
    # 클라이언트로부터 이미지 데이터 받아오기
    img = request.form['img']

    # 이미지 정보를 MongoDB에 저장
    doc = {'img': img, 'classification_result': None}  # 분류 결과는 초기에 None으로 저장
    db.test.insert_one(doc)

    return jsonify({'msg': '전송되었습니다.', 'classification_result': None})

# 서버에서 이미지를 가져오는 부분
@app.route('/api/imgdown', methods=['GET'])
def image_download():
    # 서버의 모든 이미지 리스트로 가져오기
    data = list(db.test.find({}, {'_id': False}))
    return jsonify({'downimg': data})  # 이미지 데이터 넘겨주기

# 분류결과 호출 기능 추가 코드
# 분류 결과를 생성하여 사용자에게 전송
@app.route('/api/get_classification_result', methods=['GET'])
def get_classification_result():
    # Get the most recently uploaded image
    latest_image = db.test.find_one(sort=[('_id', -1)])

    # 이미지가 존재하는지 확인
    if latest_image:
        # 이미지에 대해 분류결과가 생성되었는지 확인
        if 'classification_result' not in latest_image or latest_image['classification_result'] is None:
            img_data = latest_image.get("img").split(",")[1]  # "data:image/jpeg;base64," 부분 제거
            img_bytes = base64.b64decode(img_data)

            # BytesIO를 사용하여 이미지를 메모리에 로드
            img_buffer = BytesIO(img_bytes)

            # PIL 라이브러리를 사용하여 이미지 열기
            img = Image.open(img_buffer)
            
            # 이미지 로드 및 전처리
            img = img.resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # 모델 로드
            model = load_model(model_path)

            # 예측
            predictions = model.predict(img_array)

            # 예측에 기반하여 분류결과 결정
            if predictions[0][0] > 0.5:
                classification_result = '불법 주차 킥보드입니다.'
            else:
                classification_result = '정상 주차 킥보드입니다.'

            # DB에 있는 최근 이미지의 분류결과를 업데이트
            db.test.update_one({'_id': latest_image['_id']}, {'$set': {'classification_result': classification_result}})
        else:
            # 이미 분류결과가 생성되어있다면, 기존 결과 사용
            classification_result = latest_image['classification_result']

        return jsonify({'msg': '결과값', 'classification_result': classification_result})
    else:
        return jsonify({'msg': '이미지 없음', 'classification_result': None})

if __name__ == '__main__':
    app.run('0.0.0.0', port=5001, debug=True)



# -----------------------------------------------------------------------------------------------------------------------------------------------------
# 인공지능 모델 연결 전 코드


# from flask import Flask, render_template, jsonify, request
# from pymongo import MongoClient
# import random  # 임의의 결과를 생성하기 위해 추가


# app = Flask(__name__)

# client = MongoClient('localhost', 27017)
# db = client.dbtest


# # HTML 화면 보여주기
# @app.route('/')
# def homework():
#     return render_template('index.html')

# # 기존 코드
# # # 저장하기(POST) API
# # @app.route('/api/uploading', methods=['POST'])
# # def image_upload():
# #     # 클라이언트로 부터 정보 넘겨받기
# #     img = request.form['img']

# #     # mongodb에 이미지 넣기
# #     doc = {'img': img} # key:value
# #     db.test.insert_one(doc)

# #     return jsonify({'msg': '전송되었습니다.'})  # 저장 완료됨을 알림

# # 분류결과 호출 기능 추가 코드
# # 저장하기(POST) API
# @app.route('/api/uploading', methods=['POST'])
# def image_upload():
#     # 클라이언트로부터 이미지 데이터 받아오기
#     img = request.form['img']

#     # 이미지를 파일로 저장 (임시 저장)
#     # img_path = 'temp_image.jpg'
#     # with open(img_path, 'wb') as img_file:
#     #     img_file.write(img_data.decode('base64'))

#     # 이미지 정보를 MongoDB에 저장
#     doc = {'img': img, 'classification_result': None}  # 분류 결과는 초기에 None으로 저장
#     db.test.insert_one(doc)

#     return jsonify({'msg': '전송되었습니다.', 'classification_result': None})

# # 서버에서 이미지를 가져오는 부분
# @app.route('/api/imgdown', methods=['GET'])
# def image_download():
#     # 서버의 모든 이미지 리스트로 가져오기
#     data = list(db.test.find({}, {'_id': False}))
#     return jsonify({'downimg': data})  # 이미지 데이터 넘겨주기

# # 분류결과 호출 기능 추가 코드
# # 임의의 분류 결과를 생성하여 사용자에게 전송하는 API
# @app.route('/api/get_classification_result', methods=['GET'])
# def get_classification_result():
#     # Get the most recently uploaded image
#     latest_image = db.test.find_one(sort=[('_id', -1)])

#     # Check if the image exists
#     if latest_image:
#         # Check if the classification result is already generated for this image
#         if 'classification_result' not in latest_image or latest_image['classification_result'] is None:
#             # If not, generate a random classification result
#             random_classification_result = random.choice(['적합', '부적합'])

#             # Update the classification result for the latest image in the database
#             db.test.update_one({'_id': latest_image['_id']}, {'$set': {'classification_result': random_classification_result}})
#         else:
#             # If the classification result is already generated, use the existing result
#             random_classification_result = latest_image['classification_result']

#         return jsonify({'msg': '결과값', 'classification_result': random_classification_result})
#     else:
#         return jsonify({'msg': '이미지 없음', 'classification_result': None})

# if __name__ == '__main__':
#     app.run('0.0.0.0', port=5001, debug=True)



# -----------------------------------------------------------------------------------------------------------------------------------------------------
# 인공지능 모델 코드


# import os
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras import layers
# from tensorflow.keras import models
# from tensorflow.keras import optimizers
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# # 데이터 경로 설정
# DATA_PATH = 'Data/'

# # 이미지 크기 및 배치 크기 설정
# img_size = (224, 224)
# batch_size = 32

# # 이미지 데이터 증강을 위한 설정
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     brightness_range=[0.5, 1.5],  # 밝기 조절 추가
#     horizontal_flip=True,
#     fill_mode='nearest',
#     validation_split=0.2
# )

# # 이미지 데이터를 불러와서 증강
# train_generator = train_datagen.flow_from_directory(
#     DATA_PATH,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode='binary',
#     subset='training'
# )

# validation_generator = train_datagen.flow_from_directory(
#     DATA_PATH,
#     target_size=img_size,
#     batch_size=batch_size,
#     class_mode='binary',
#     subset='validation'
# )

# # VGG16 모델 불러오기
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# # 새로운 모델 정의
# model = models.Sequential()
# model.add(base_model)
# model.add(layers.GlobalAveragePooling2D())
# model.add(layers.Dense(256, activation='relu', kernel_initializer='he_normal'))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(128, activation='relu', kernel_initializer='he_normal'))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(1, activation='sigmoid'))

# # VGG16의 가중치를 동결
# base_model.trainable = False

# # 모델 컴파일
# total_samples = train_generator.samples
# weight_for_0 = total_samples / (2 * 77)
# weight_for_1 = total_samples / (2 * 172)

# class_weight = {0: weight_for_0, 1: weight_for_1}

# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.RMSprop(lr=1e-6),  # 더 작은 학습률 시도
#               metrics=['accuracy'])

# # 조기 종료와 모델 체크포인트 설정
# early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
# checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# # 모델 학습
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // batch_size,
#     epochs=1000,  # 더 많은 Epoch 시도
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // batch_size,
#     callbacks=[early_stopping, checkpoint],
#     class_weight=class_weight
# )

# # 학습된 모델 저장
# model.save('DSC/vgg16_transfer_learning.h5')