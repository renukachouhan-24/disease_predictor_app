import os
import json
import requests
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, session, abort
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from oauthlib.oauth2 import WebApplicationClient
from dotenv import load_dotenv

# Environment settings
load_dotenv()
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1' # Only for local testing

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "any_random_string")

# Google Configuration
GOOGLE_CLIENT_ID = os.getenv("CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("CLIENT_SECRET")
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

# OAuth 2.0 Client Setup
client = WebApplicationClient(GOOGLE_CLIENT_ID)

# Flask-Login Setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

class User(UserMixin):
    def __init__(self, id_, name, email, profile_pic):
        self.id = id_
        self.name = name
        self.email = email
        self.profile_pic = profile_pic

# Dummy user storage (Production mein database use hota hai)
users_db = {}

@login_manager.user_loader
def load_user(user_id):
    return users_db.get(user_id)

def get_google_provider_cfg():
    return requests.get(GOOGLE_DISCOVERY_URL).json()

# ---------------- AI Models Loading ----------------
MODELS = {
    'lung': load_model('models/lung_model.h5'),
    'ovarian': load_model('models/ovarian_model.h5'),
    'pancreatic': load_model('models/pancreatic_model.h5')
}

with open('models/lung_labels.json') as f: lung_labels = json.load(f)
with open('models/ovarian_labels.json') as f: ovarian_labels = json.load(f)
with open('models/pancreatic_labels.json') as f: pancreatic_labels = json.load(f)

LABELS = {'lung': lung_labels, 'ovarian': ovarian_labels, 'pancreatic': pancreatic_labels}

FRIENDLY_NAMES = {
    'Adenocarcinoma': 'Cancer Detected (Adenocarcinoma)',
    'Squamous_Cell': 'Cancer Detected (Squamous Cell Carcinoma)',
    'Normal': 'Normal (No Cancer Detected)',
    'MC': 'Cancer Detected (Mucinous Carcinoma)',
    'HGSC': 'Cancer Detected (High-Grade Serous Carcinoma)',
    'Tumor': 'Cancer Detected (Pancreatic Tumor)',
}

# ---------------- Authentication Routes ----------------

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/login/google")
def google_login():
    google_provider_cfg = get_google_provider_cfg()
    authorization_endpoint = google_provider_cfg["authorization_endpoint"]

    request_uri = client.prepare_request_uri(
        authorization_endpoint,
        redirect_uri=request.base_url.replace("http://", "http://") + "/callback",
        scope=["openid", "email", "profile"],
    )
    return redirect(request_uri)

@app.route("/login/google/callback")
def callback():
    code = request.args.get("code")
    google_provider_cfg = get_google_provider_cfg()
    token_endpoint = google_provider_cfg["token_endpoint"]

    token_url, headers, body = client.prepare_token_request(
        token_endpoint,
        authorization_response=request.url,
        redirect_url=request.base_url,
        code=code
    )
    token_response = requests.post(
        token_url,
        headers=headers,
        data=body,
        auth=(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET),
    )

    client.parse_request_body_response(json.dumps(token_response.json()))

    userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]
    uri, headers, body = client.add_token(userinfo_endpoint)
    userinfo_response = requests.get(uri, headers=headers, data=body)

    if userinfo_response.json().get("email_verified"):
        unique_id = userinfo_response.json()["sub"]
        users_email = userinfo_response.json()["email"]
        picture = userinfo_response.json()["picture"]
        users_name = userinfo_response.json()["given_name"]
    else:
        return "User email not available or not verified by Google.", 400

    user = User(id_=unique_id, name=users_name, email=users_email, profile_pic=picture)
    users_db[unique_id] = user

    login_user(user)
    return redirect(url_for("home"))

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ---------------- Prediction Routes ----------------

@app.route('/')
@login_required
def home():
    return render_template('index.html', user_name=current_user.name)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        disease = request.form.get('disease')
        img_file = request.files['image']
        
        if not img_file: return "No image uploaded", 400

        if not os.path.exists('static/uploads'): os.makedirs('static/uploads')
            
        img_path = os.path.join("static/uploads", img_file.filename)
        img_file.save(img_path)
        
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = MODELS[disease].predict(img_array)
        class_idx = np.argmax(prediction[0])
        
        tech_name = LABELS[disease].get(str(class_idx), "Unknown")
        final_result = FRIENDLY_NAMES.get(tech_name, tech_name)

        return render_template('index.html', prediction_text=final_result, disease=disease)
                               
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)