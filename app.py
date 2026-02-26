import os
import json
import requests
import numpy as np
import tensorflow as tf
import h5py
from flask import Flask, request, render_template, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from tensorflow.keras.preprocessing import image
from oauthlib.oauth2 import WebApplicationClient
from dotenv import load_dotenv
import io
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1' 
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "any_random_string")

# Google Configuration
GOOGLE_CLIENT_ID = os.getenv("CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("CLIENT_SECRET")
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"
client = WebApplicationClient(GOOGLE_CLIENT_ID)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

class User(UserMixin):
    def __init__(self, id_, name, email, profile_pic):
        self.id = id_; self.name = name; self.email = email; self.profile_pic = profile_pic

users_db = {}
@login_manager.user_loader
def load_user(user_id): return users_db.get(user_id)

def get_google_provider_cfg(): return requests.get(GOOGLE_DISCOVERY_URL).json()

FRIENDLY_NAMES = {
    'Bengin cases': 'Normal (Benign)',
    'Malignant cases': 'Cancer Detected (Malignant)',
    'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib': 'Cancer Detected (Adenocarcinoma)',
    'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa': 'Cancer Detected (Large Cell)',
    'normal': 'Normal (No Cancer)',
    'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa': 'Cancer Detected (Squamous Cell)',
    'MC': 'Cancer Detected (Mucinous)', 
    'HGSC': 'Cancer Detected (Serous)', 
    'Tumor': 'Pancreatic Tumor',
}

def load_lung_model_safe(model_path):
    """
    Safe loader for corrupted lung model - loads weights only and rebuilds architecture
    """
    import h5py
    
    with h5py.File(model_path, 'r') as f:
        try:
            optimizer_data = f['optimizer_weights']['adam']
            dense_3_bias_key = 'sequential_1_dense_3_bias_momentum'
            if dense_3_bias_key in optimizer_data:
                num_classes = optimizer_data[dense_3_bias_key].shape[0]
            else:
                num_classes = 6  # Default for lung cancer
                
            # Get dense_2 units
            dense_2_bias_key = 'sequential_1_dense_2_bias_momentum'
            if dense_2_bias_key in optimizer_data:
                dense_units = optimizer_data[dense_2_bias_key].shape[0]
            else:
                dense_units = 256  # Default
                
        except Exception as e:
            print(f"Using defaults: {e}")
            num_classes = 6
            dense_units = 256
    
    print(f"Building model: {dense_units} units → {num_classes} classes")
    
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False, 
        weights=None, 
        input_shape=(224, 224, 3),
        name='mobilenetv2_1.00_224'
    )
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d_1'),
        tf.keras.layers.Dense(dense_units, activation='relu', name='dense_2'),
        tf.keras.layers.Dropout(0.3, name='dropout_1'),
        tf.keras.layers.Dense(num_classes, activation='softmax', name='dense_3')
    ], name='sequential_1')
    
    model.build((None, 224, 224, 3))
    
    try:
        model.load_weights(model_path)
        print(f"✓ Successfully loaded lung model weights")
    except Exception as e:
        print(f"⚠ Warning loading weights: {e}")
        try:
            model.load_weights(model_path, skip_mismatch=True)
            print(f"✓ Loaded weights with skip_mismatch=True")
        except Exception as e2:
            print(f"❌ Could not load weights: {e2}")
            return None
    
    return model

@app.route("/login")
def login(): return render_template("login.html")

@app.route("/login/google")
def google_login():
    google_provider_cfg = get_google_provider_cfg()
    authorization_endpoint = google_provider_cfg["authorization_endpoint"]
    request_uri = client.prepare_request_uri(
        authorization_endpoint, redirect_uri=request.base_url + "/callback",
        scope=["openid", "email", "profile"],
    )
    return redirect(request_uri)

@app.route("/login/google/callback")
def callback():
    try:
        code = request.args.get("code")
        google_provider_cfg = get_google_provider_cfg()
        token_endpoint = google_provider_cfg["token_endpoint"]
        token_url, headers, body = client.prepare_token_request(
            token_endpoint, authorization_response=request.url,
            redirect_url=request.base_url, code=code
        )
        token_response = requests.post(token_url, headers=headers, data=body, auth=(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET))
        client.parse_request_body_response(json.dumps(token_response.json()))
        userinfo_endpoint = google_provider_cfg["userinfo_endpoint"]
        uri, headers, body = client.add_token(userinfo_endpoint)
        userinfo_response = requests.get(uri, headers=headers, data=body).json()
        
        unique_id = userinfo_response["sub"]
        user = User(id_=unique_id, name=userinfo_response["given_name"], email=userinfo_response["email"], profile_pic=userinfo_response["picture"])
        users_db[unique_id] = user
        login_user(user)
        return redirect(url_for("home"))
    except Exception as e: return f"Error: {e}"

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route('/')
@login_required
def home(): return render_template('index.html', user_name=current_user.name)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    tf.keras.backend.clear_session()
    import gc
    gc.collect()
    
    try:
        disease = request.form.get('disease')
        img_file = request.files['image']
        
        if not img_file:
            return render_template('index.html', prediction_text="Error: No image uploaded", user_name=current_user.name)

        label_path = f'models/{disease}_labels.json'
        with open(label_path) as f: 
            current_labels = json.load(f)

        img = Image.open(io.BytesIO(img_file.read())).convert('RGB').resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  

        model_path = f'models/{disease}_models.h5'
        
        if disease == 'lung':
            model = load_lung_model_safe(model_path)
        else:
            model = tf.keras.models.load_model(model_path, compile=False)
        
        predictions = model.predict(img_array, verbose=0)
        
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)

        if confidence < 30:
            final_result = f"⚠️ Unclear Scan ({confidence:.1f}% confidence). Please ensure correct {disease.capitalize()} scan."
        else:
            tech_name = current_labels.get(str(class_idx), "Unknown")
            final_result = f"{FRIENDLY_NAMES.get(tech_name, tech_name)} ({confidence:.1f}%)"

        del model
        tf.keras.backend.clear_session()
        gc.collect()
        
        return render_template('index.html', prediction_text=final_result, confidence_score=confidence, user_name=current_user.name)

    except Exception as e:
        tf.keras.backend.clear_session()
        import gc
        gc.collect()
        return render_template('index.html', prediction_text=f"Error: {str(e)}", user_name=current_user.name)
    
if __name__ == "__main__":
    app.run(debug=True, threaded=False)