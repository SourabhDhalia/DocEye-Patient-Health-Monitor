from flask import Flask, render_template, url_for, redirect, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_bcrypt import Bcrypt
import pickle
import cv2
# from camera import VideoCamera

import numpy as np
# Dlib for deep learning based Modules and face landmark detection
import dlib
#face_utils for basic operations of conversion
from imutils import face_utils

# Twilio API
from email import message
from http import client
from twilio.rest import Client
import keys

import time
app = Flask(__name__)
camera = cv2.VideoCapture("templates\9.mp4")
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'
first_frame=None
video_camera = None
global_frame = None


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class RegisterForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')


class LoginForm(FlaskForm):
    username = StringField(validators=[
                           InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})

    password = PasswordField(validators=[
                             InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            detector=cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
            faces=detector.detectMultiScale(frame,1.1,7)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
             #Draw the rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break


# model =pickle
@app.route('/')
def home():
    return render_template('home.html')

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
cap = cv2.VideoCapture("templates\9.mp4")

    # frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')

out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280,720))
client=Client(keys.account_sid,keys.auth_token)
@app.route('/#')

def msg2():
	message=client.messages.create(
		body="!!!!!!!!!!!!!!!---Help---Emergency Alert---!!!!!!!!!!!!!! ",
		from_=keys.twilio_number,
		to=keys.my_phone_number  
	)
	return message
def sd_motion():
    val=0
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    print(frame1.shape)
    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) < 100:
                continue
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)
            val+=1
            print(val)
            if val==76:
                break
        cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

        image = cv2.resize(frame1, (1280,720))
        out.write(image)
        cv2.imshow("feed", frame1)
        frame1 = frame2
        ret, frame2 = cap.read()
        # if val==50000:
        #     break
        if val==76:
            msg2()
            break
        if cv2.waitKey(50) == 27:
            break
        # time.sleep(1)
        # val+=1
    cv2.destroyAllWindows()
    cap.release()
    out.release()


# ////////////////////////////////////////////////////////////////////////////////////////////////

# *************************************************************************************************

#Initializing the camera and taking the instance
# cap = cv2.VideoCapture(0)

#Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
client=Client(keys.account_sid,keys.auth_token)

# #Initializing the face detector and landmark detector
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
@app.route('/##')
def msg():
	message=client.messages.create(
		body="!!!!!!!!!!!!!!!---Help---Emergency Alert---!!!!!!!!!!!!!! ",
		from_=keys.twilio_number,
		to=keys.my_phone_number  
	)
	return message


def compute(ptA,ptB):
	dist = np.linalg.norm(ptA - ptB)
	return dist

def blinked(a,b,c,d,e,f):
	up = compute(b,d) + compute(c,e)
	down = compute(a,f)
	ratio = up/(2.0*down)

	#Checking if it is blinked
	if(ratio>0.25):
		return 2
	elif(ratio>0.21 and ratio<=0.25):
		return 1
	else:
		return 0

def sd_eyes(): 
    sleep = 0
    drowsy = 0
    active = 0
    status=""
    color=(0,0,0)
    # print("hii")
    
    while True:
        _, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
	    #detected face in faces array
        for face in faces:
	        x1 = face.left()
	        y1 = face.top()
	        x2 = face.right()
	        y2 = face.bottom()

	        face_frame = frame.copy()
	        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

	        landmarks = predictor(gray, face)
	        landmarks = face_utils.shape_to_np(landmarks)

	        #The numbers are actually the landmarks which will show eye
	        left_blink = blinked(landmarks[36],landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
	        right_blink = blinked(landmarks[42],landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])
	
	        #Now judge what to do for the eye blinks
	        if(left_blink==0 or right_blink==0):
	        	sleep+=1
	        	drowsy=0
	        	active=0
	        	if(sleep>6):
	        		status="SLEEPING !!!"
	        		color = (255,0,0)

	        elif(left_blink==1 or right_blink==1):
	        	sleep=0
	        	active=0
	        	drowsy+=1
	        	if(drowsy>6):
	        		status="Drowsy !!!"
	        		color = (0,0,255)

	        else:
	        	drowsy=0
	        	sleep=0
	        	active+=1
	        	if(active>6):
	        		status=msg()
	        		color = (0,255,0)
	
	        cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,3)

	        for n in range(0, 68):
	        	(x,y) = landmarks[n]
	        	cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

        cv2.imshow("Frame", frame)
        cv2.imshow("Result of detector", face_frame)
        key = cv2.waitKey(1)
        if key == 27:
           break



# *************************************************************************************************

@app.route('/sd_motion2')
def sd_motion2():
    return Response(sd_motion(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/Face_model')
def indexML1():
    return render_template('indexML1.html')
@app.route('/video_feed')
def video_feed():
    return Response(sd_eyes(), mimetype='multipart/x-mixed-replace; boundary=frame')


# MOdel 2 Motion detection
@app.route('/Motion_detection')
def indexML2():
    return render_template('indexML2.html')

@app.route('/record_status', methods=['POST'])
def record_status():
    global video_camera 
    if video_camera == None:
        video_camera = VideoCamera()

    json = request.get_json()

    status = json['status']

    if status == "true":
        video_camera.start_record()
        return jsonify(result="started")
    else:
        video_camera.stop_record()
        return jsonify(result="stopped")



def video_stream():
    global video_camera 
    global global_frame

    if video_camera == None:
        video_camera = VideoCamera()
        
    while True:
        frame = video_camera.get_frame()
        if frame != None:
            global_frame = frame
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')

@app.route('/video_viewer')
def video_viewer():
    return Response(sd_motion(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')





@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))
    return render_template('login.html', form=form)


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('dashboard.html')


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@ app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html', form=form)


if __name__ == "__main__":
    app.run(debug=True)
