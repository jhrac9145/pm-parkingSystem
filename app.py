from flask import Flask, render_template, jsonify, request
from pymongo import MongoClient
import random  # 임의의 결과를 생성하기 위해 추가


app = Flask(__name__)

client = MongoClient('localhost', 27017)
db = client.dbtest


# HTML 화면 보여주기
@app.route('/')
def homework():
    return render_template('index.html')

# 기존 코드
# # 저장하기(POST) API
# @app.route('/api/uploading', methods=['POST'])
# def image_upload():
#     # 클라이언트로 부터 정보 넘겨받기
#     img = request.form['img']

#     # mongodb에 이미지 넣기
#     doc = {'img': img} # key:value
#     db.test.insert_one(doc)

#     return jsonify({'msg': '전송되었습니다.'})  # 저장 완료됨을 알림

# 분류결과 호출 기능 추가 코드
# 저장하기(POST) API
@app.route('/api/uploading', methods=['POST'])
def image_upload():
    # 클라이언트로부터 이미지 데이터 받아오기
    img = request.form['img']

    # 이미지를 파일로 저장 (임시 저장)
    # img_path = 'temp_image.jpg'
    # with open(img_path, 'wb') as img_file:
    #     img_file.write(img_data.decode('base64'))

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
# 임의의 분류 결과를 생성하여 사용자에게 전송하는 API
@app.route('/api/get_classification_result', methods=['GET'])
def get_classification_result():
    # Get the most recently uploaded image
    latest_image = db.test.find_one(sort=[('_id', -1)])

    # Check if the image exists
    if latest_image:
        # Check if the classification result is already generated for this image
        if 'classification_result' not in latest_image or latest_image['classification_result'] is None:
            # If not, generate a random classification result
            random_classification_result = random.choice(['적합', '부적합'])

            # Update the classification result for the latest image in the database
            db.test.update_one({'_id': latest_image['_id']}, {'$set': {'classification_result': random_classification_result}})
        else:
            # If the classification result is already generated, use the existing result
            random_classification_result = latest_image['classification_result']

        return jsonify({'msg': '결과값', 'classification_result': random_classification_result})
    else:
        return jsonify({'msg': '이미지 없음', 'classification_result': None})

if __name__ == '__main__':
    app.run('0.0.0.0', port=5001, debug=True)