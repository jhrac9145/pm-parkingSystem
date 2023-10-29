from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client.dbtest


# HTML 화면 보여주기
@app.route('/')
def homework():
    return render_template('index.html')


# 저장하기(POST) API
@app.route('/api/uploading', methods=['POST'])
def image_upload():
    # 클라이언트로 부터 정보 넘겨받기
    img = request.form['img']
    # img = request.form.get('img')

    # mongodb에 이미지 넣기
    doc = {'img': img} # key:value
    db.test.insert_one(doc)

    return jsonify({'msg': '저장되었습니다.'})  # 저장 완료됨을 알림


# 서버에서 이미지를 가져오는 부분
@app.route('/api/imgdown', methods=['GET'])
def image_download():
    # 서버의 모든 이미지 리스트로 가져오기
    data = list(db.test.find({}, {'_id': False}))
    return jsonify({'downimg': data})  # 이미지 데이터 넘겨주기

if __name__ == '__main__':
    app.run('0.0.0.0', port=5001, debug=True)