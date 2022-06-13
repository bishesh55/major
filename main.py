from flask import Flask, render_template, request, Response, redirect
import cv2
import numpy as np
from keras.models import load_model
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
video = cv2.VideoCapture(0)


app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template('index.html')




def gen():
    """Video streaming generator function."""
    while True:
        rval, frame = video.read()
        cv2.imwrite('t.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    request.form['btn']
    _, frame = video.read()
    cv2.imwrite('static/file.jpg', frame)
    img1 = cv2.imread('static/file.jpg')
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    faces = cascade.detectMultiScale(gray, 1.1, 3)
    num = len(faces)
    if num == 1:

        for x, y, w, h in faces:
            cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cropped = img1[y:y + h, x:x + w]

        cv2.imwrite('static/after.jpg', img1)
        try:
            cv2.imwrite('static/cropped.jpg', cropped)

        except:
            pass

        try:
            img = cv2.imread('static/cropped.jpg', 0)

        except:
            img = cv2.imread('static/file.jpg', 0)

        img = cv2.resize(img, (48, 48))
        img = img / 255

        img = img.reshape(1, 48, 48, 1)

        model = load_model('model.h5')

        pred = model.predict(img)

        label_map = ['Anger', 'Neutral', 'Fear', 'Happy', 'Sad', 'Surprise']
        pred = np.argmax(pred)
        final_pred = label_map[pred]
        allTodo = Todo.query.filter_by(mood=final_pred)

        return render_template('predict.html', data=final_pred, allTodo=allTodo)
    else:
        return render_template('error.html')

app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///todo.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Todo(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    artist = db.Column(db.String(200), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    link = db.Column(db.String(500), nullable=False)
    mood = db.Column(db.String(200), nullable=False)

    def __repr__(self) -> str:
        return f"{self.sno} - {self.title} - {self.artist} - {self.year} - {self.link} - {self.mood}"

@app.route('/admin', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        title = request.form['title']
        artist = request.form['artist']
        year = request.form['year']
        link = request.form['link']
        mood = request.form['mood']

        todo = Todo(title=title, artist=artist, year=year, link=link, mood=mood)
        db.session.add(todo)
        db.session.commit()

    allTodo = Todo.query.all()
    return render_template('indexb.html', allTodo=allTodo)

    # return render_template('indexb.html')

@app.route('/delete/<int:sno>')
def delete(sno):
    todo = Todo.query.filter_by(sno=sno).first()
    db.session.delete(todo)
    db.session.commit()
    return redirect("/admin")

if __name__ == "__main__":
    app.run(debug=True)