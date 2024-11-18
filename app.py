import os
from flask import Flask, request, redirect, render_template, flash, Blueprint
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import cv2

classes = ["ダンボール","古紙","古着","包装プラスチック類","ペットボトル類","ライター","ビン類","生ゴミ","空き缶","紙オムツ","蛍光灯","電池・バッテリー","靴","衛生用品","汚れた紙類"]
image_size = 64

categories = {"燃やせるごみ": [classes[14], classes[13], classes[12], classes[9], classes[7]],"古紙ごみ": [classes[0], classes[1]],"梱包包装プラスチックごみ": [classes[3]],"古着ごみ": [classes[2]],"ペットボトルごみ": [classes[4]], "有害物ごみ": [classes[5], classes[11], classes[10]],"缶ごみ": [classes[8]],"瓶ごみ": [classes[6]],}

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif','webp'])

app = Flask(__name__)
app.secret_key = 'abcdefzyxdls'  # シークレットキーを設定

add_app = Blueprint("images", __name__, static_url_path="/images", static_folder="./images")
app.register_blueprint(add_app)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model('./model.h5',compile=False)  # 学習済みモデルをロード


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルを入力してください。')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ゴミはきちんと分別しましょう。')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path=os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            img=cv2.imread(file_path)

            img = cv2.resize(img,(image_size,image_size))
            img = image.img_to_array(img)
            data = np.array([img])
            data = data.astype("float")/255
            data = data.reshape(1,64,64,3)

            result = model.predict(data)
            predicted = np.argmax(result)

            pred=classes[predicted]
            pred_answer=pred
            for category, items in categories.items():
                if pred_answer in items:
                    answer =("おそらく{}なので、「{}」です。".format(classes[predicted],category))

                    return render_template("index.html", answer=answer)
        else:
            flash('許可されていない拡張子です。')
            return redirect(request.url)\

    return render_template("index.html", answer="")



if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)
