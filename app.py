import os
import json
import requests
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from oauthlib.oauth2 import WebApplicationClient
from dotenv import load_dotenv
import io
from PIL import Image

# Environment settings
load_dotenv()
# Render par HTTPS zaroori hai, local ke liye insecure allow karte hain
if os.getenv('RENDER'):
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '0'
else:
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "any_random_string")

# Google Configuration
GOOGLE_CLIENT_ID = os.getenv("CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("CLIENT_SECRET")
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

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

users_db = {}

@login_manager.user_loader
def load_user(user_id):
    return users_db.get(user_id)

def get_google_provider_cfg():
    return requests.get(GOOGLE_DISCOVERY_URL).json()

# Friendly names logic
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
        redirect_uri=request.base_url + "/callback",
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
        token_url, headers=headers, data=body,
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
        
        user = User(id_=unique_id, name=users_name, email=users_email, profile_pic=picture)
        users_db[unique_id] = user
        login_user(user)
        return redirect(url_for("home"))
    return "User email not verified.", 400

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ---------------- Prediction Routes (Optimized) ----------------

@app.route('/')
@login_required
def home():
    return render_template('index.html', user_name=current_user.name)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    model = None
    try:
        disease = request.form.get('disease')
        img_file = request.files['image']
        
        if not img_file: return "No image uploaded", 400

        # Memory management: Load labels locally
        label_path = f'models/{disease}_labels.json'
        with open(label_path) as f:
            current_labels = json.load(f)

        # Image processing in memory (Bina save kiye)
        img_content = img_file.read()
        img = Image.open(io.BytesIO(img_content)).convert('RGB')
        img = img.resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # LAZY LOADING: Model sirf tabhi load hoga jab zaroorat ho
        model_path = f'models/{disease}_model.h5'
        model = load_model(model_path)
        
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction[0])
        
        tech_name = current_labels.get(str(class_idx), "Unknown")
        final_result = FRIENDLY_NAMES.get(tech_name, tech_name)

        # Cleaning memory after prediction
        del model
        tf.keras.backend.clear_session()

        return render_template('index.html', 
                               prediction_text=final_result, 
                               disease=disease,
                               user_name=current_user.name)
                               
    except Exception as e:
        tf.keras.backend.clear_session()
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)