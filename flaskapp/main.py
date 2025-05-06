from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import warnings
import pickle
import xgboost as xgb
import sys
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

def debug_print(*args):
    print(*args, file=sys.stderr)

def load_model(file_path):
    filename = os.path.basename(file_path)
    try:
        if "xgboost" in filename.lower() and file_path.endswith('.model'):
            model = xgb.Booster()
            model.load_model(file_path)
            return model
        else:
            return joblib.load(file_path)
    except Exception as e:
        debug_print(f"Failed to load {filename}: {e}")
        return None

# Load models
models = {}
model_dir = r"D:\3rd year even sem\Term Paper\Project\models"

if not os.path.exists(model_dir):
    debug_print(f"WARNING: Model directory does not exist: {model_dir}")
    model_dir = os.path.join(os.getcwd(), "models")
    debug_print(f"Trying alternative path: {model_dir}")

if os.path.exists(model_dir):
    debug_print(f"Loading models from: {model_dir}")
    for filename in os.listdir(model_dir):
        if filename.endswith(".pkl"):
            if "decision_tree" in filename or "random_forest" in filename:
                debug_print(f"Skipping known incompatible model: {filename}")
                continue
            file_path = os.path.join(model_dir, filename)
            try:
                models[filename.replace(".pkl", "")] = joblib.load(file_path)
                debug_print(f"Successfully loaded: {filename}")
            except Exception as e:
                debug_print(f"Failed to load {filename}: {e}")
else:
    debug_print("No models directory found. Please ensure the models directory exists.")
    models["test_model"] = "dummy_model"

def preprocess_input(form_data):
    sleep_map = {
        "Less than 5 hours": 0,
        "5-6 hours": 1,
        "7-8 hours": 2,
        "More than 8 hours": 3
    }
    dietary_map = {"Unhealthy": 0, "Moderate": 1, "Healthy": 2}
    suicidal_map = {"No": 0, "Yes": 1, "Sometimes": 2, "Often": 3}
    family_history_map = {"No": 0, "Yes": 1}

    age = int(form_data.get("age", 0))
    academic_pressure = int(form_data.get("academic_pressure", 0))
    work_pressure = int(form_data.get("work_pressure", 0))
    cgpa = float(form_data.get("cgpa", 0))
    study_satisfaction = int(form_data.get("study_satisfaction", 0))
    job_satisfaction = int(form_data.get("job_satisfaction", 0))
    sleep_duration_raw = form_data.get("sleep_duration", "7-8 hours").replace("'", "").strip()
    sleep_duration = sleep_map.get(sleep_duration_raw, 7.0)
    dietary_habits = dietary_map.get(form_data.get("dietary_habits", "Moderate"), 1)
    suicidal_thoughts = suicidal_map.get(form_data.get("suicidal_thoughts", "No"), 0)
    work_study_hours = float(form_data.get("work_study_hours", 0))
    financial_stress = int(form_data.get("financial_stress", 0))
    family_history = family_history_map.get(form_data.get("family_history", "No"), 0)

    inputs = [
        age, academic_pressure, work_pressure, cgpa,
        study_satisfaction, job_satisfaction,
        sleep_duration, dietary_habits,
        suicidal_thoughts, work_study_hours,
        financial_stress, family_history
    ]
    print(inputs)
    return np.array(inputs).reshape(1, -1)

# 💖 Comfort Content
def get_soothing_music():
    url = 'https://www.youtube.com/results?search_query=soothing+calm+music+playlist'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    results = [a.text for a in soup.select('a#video-title')][:5]
    return results

def get_soothing_movies():
    return [
        "The Secret Life of Walter Mitty",
        "Amélie",
        "Inside Out",
        "My Neighbor Totoro",
        "The Pursuit of Happyness"
    ]

def get_mental_health_remedies():
    return [
        "Practice daily mindfulness meditation 🧘‍♀️",
        "Establish a consistent sleep schedule 😴",
        "Stay hydrated and eat balanced meals 🥗",
        "Talk to someone you trust 💬",
        "Take regular breaks and get fresh air 🍃"
    ]

def get_healing_quotes():
    return [
        "You don’t have to control your thoughts. You just have to stop letting them control you. — Dan Millman",
        "This too shall pass 🌈",
        "Healing takes time, and that's okay, love 💖",
        "Sometimes the bravest thing you can do is ask for help 🌱"
    ]

def get_healing_hobbies():
    return [
        "Painting or sketching 🎨",
        "Listening to soft instrumental music 🎧",
        "Journaling your thoughts 🖋️",
        "Practicing gentle yoga 🧘‍♀️",
        "Gardening and tending to plants 🌿"
    ]

def get_healing_activities():
    return [
        "Take a warm bath 🛁",
        "Do deep breathing exercises 🌬️",
        "Light a candle and listen to rain sounds 🕯️🌧️",
        "Declutter a corner of your space 🧹",
        "Lie down and imagine your happy place ☁️"
    ]

# 🌞 Cheerful Content
def get_positive_habits():
    return [
        "Start a gratitude journal 🌞",
        "Take a short walk every evening 🚶‍♂️",
        "Read an uplifting book 📖",
        "Compliment someone today 💌",
        "Unplug from tech for 1 hour 🔌"
    ]

def get_happy_books():
    return [
        "The Alchemist",
        "Ikigai",
        "Big Magic",
        "Atomic Habits",
        "Tuesdays with Morrie"
    ]

def get_happy_music():
    return [
        "Happy – Pharrell Williams",
        "Good Life – OneRepublic",
        "Pocketful of Sunshine – Natasha Bedingfield",
        "Count on Me – Bruno Mars"
    ]

def get_cheerful_quotes():
    return [
        "Believe you can and you're halfway there. — Theodore Roosevelt ☀️",
        "You are stronger than you think and more loved than you know 💪💕",
        "Happiness is not by chance, but by choice 🌼",
        "Keep your face to the sunshine and you cannot see a shadow ☀️"
    ]

def get_cheerful_hobbies():
    return [
        "Dancing your heart out 💃",
        "Trying a fun DIY project ✂️",
        "Singing in the shower 🎤",
        "Exploring nature walks 🌳",
        "Photography adventures 📸"
    ]

def get_cheerful_activities():
    return [
        "Bake cookies or something sweet 🍪",
        "Call a friend and laugh a little 😂",
        "Watch funny animal videos 🐶🐱",
        "Create a vision board 🧡",
        "Plan a little weekend adventure ✨"
    ]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    selected_model = None
    error_message = None
    music = movies = remedies = positive = books = happy_songs = None
    quotes = hobbies = activities = None

    if request.method == "POST":
        selected_model = request.form.get("model", "xgboost_gpu")
        try:
            input_array = preprocess_input(request.form)
            debug_print(f"Input array: {input_array}")
            model = models.get(selected_model)
            debug_print(f"Model type: {type(model)}")

            if model:
                if model == "dummy_model":
                    prediction = "Test prediction"
                    debug_print("Using dummy model for testing")
                else:
                    prediction = model.predict(input_array)[0]

                    if prediction == 1:
                        music = get_soothing_music()
                        movies = get_soothing_movies()
                        remedies = get_mental_health_remedies()
                        quotes = get_healing_quotes()
                        hobbies = get_healing_hobbies()
                        activities = get_healing_activities()
                        positive = books = happy_songs = None
                    else:
                        remedies = music = movies = None
                        quotes = get_cheerful_quotes()
                        hobbies = get_cheerful_hobbies()
                        activities = get_cheerful_activities()
                        positive = get_positive_habits()
                        books = get_happy_books()
                        happy_songs = get_happy_music()

                debug_print(f"Prediction result: {prediction}")
            else:
                error_message = f"Model '{selected_model}' not loaded properly."
                debug_print(error_message)
        except Exception as e:
            error_message = f"Error making prediction: {str(e)}"
            debug_print(f"Exception: {str(e)}")
            import traceback
            debug_print(traceback.format_exc())

    return render_template('form.html', prediction=prediction,
                           musics=music, movies=movies, remedies=remedies,
                           positive=positive, books=books, happy_songs=happy_songs,
                           quotes=quotes, hobbies=hobbies, activities=activities)

if __name__ == "__main__":
    debug_print("Starting Flask application... 💫")
    app.run(debug=True)
