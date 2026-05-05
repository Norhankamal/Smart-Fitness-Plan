#####   Flask    ######
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

app = Flask(__name__)

# Load saved models and encoders
clf = joblib.load('models/fitness_goal_classifier.pkl')  # Load your saved classifier model
regressor = joblib.load('models/calories_burned_regressor.pkl')  # Load your saved regressor model
scaler = joblib.load('models/scaler.pkl')  # Load scaler used for normalizing data
label_encoders = joblib.load('models/label_encoders.pkl')  # Load label encoders

# Features for classification and regression
level1_features = ['Age', 'Gender', 'BMI', 'Workout_Frequency (days/week)',
                   'Experience_Level', 'Workout_Type', 'Fat_Percentage']
level2_features = ['Age', 'Gender', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM',
                   'Session_Duration (hours)', 'Workout_Type', 'Fat_Percentage', 'BMI', 'Predicted_Fitness_Goal']

# Define optimization function for Level 3 with dynamic adjustments
def objective(workout_plan, alpha=0.6, beta=0.3, gamma=0.1, user_experience=1, predicted_calories=1, fitness_goal=None):
    session_duration, target_bpm, fatigue_risk = workout_plan
    adjusted_alpha = alpha * user_experience
    adjusted_beta = beta * (target_bpm / 200)
    adjusted_gamma = gamma * (1 + fatigue_risk)

    # Adjust goal alignment based on predicted fitness goal
    goal_alignment = 0
    if fitness_goal == 'Weight_Loss':
        goal_alignment = 1  # prioritize higher calories burnt for weight loss
    elif fitness_goal == 'Muscle_Gain':
        goal_alignment = 0.8  # slightly less focus on calories burnt
    else:
        goal_alignment = 0.9  # balance for endurance

    # Add penalty for long session durations if user has low experience
    penalty_duration = 0 if session_duration <= 1 + user_experience * 0.2 else session_duration * 0.5

    return -(adjusted_alpha * predicted_calories * goal_alignment - adjusted_beta - adjusted_gamma + penalty_duration)

def constraint_duration(workout_plan, user_experience):
    session_duration = workout_plan[0]
    max_duration = 0.5 + user_experience * 0.3
    return max_duration - session_duration

def constraint_bpm(workout_plan, max_bpm, age):
    target_bpm = workout_plan[1]
    safe_bpm = max_bpm - (0.5 * (age - 30))
    return safe_bpm - target_bpm

def get_initial_plan(user_experience):
    return [np.random.uniform(0.5, 2.0), np.random.uniform(100, 180), np.random.uniform(0.0, 1.0)]

bounds = [(0.5, 2.0), (100, 200), (0.0, 1.0)]

# Function to calculate the optimal plan based on user input
def calculate_optimal_plan(user_experience=1, max_bpm=200, predicted_calories=1, age=30, fitness_goal=None):
    initial_plan = get_initial_plan(user_experience)
    result = minimize(objective, initial_plan, bounds=bounds,
                      constraints=[
                          {"type": "ineq", "fun": lambda workout_plan: constraint_duration(workout_plan, user_experience)},
                          {"type": "ineq", "fun": lambda workout_plan: constraint_bpm(workout_plan, max_bpm, age)}
                      ], args=(0.6, 0.3, 0.1, user_experience, predicted_calories, fitness_goal))
    return result.x  # Return the optimal workout plan

# Function to calculate water intake
def calculate_water_intake(weight, session_duration):
    base_water = 0.03 * weight
    extra_water = session_duration * 0.5
    return base_water + extra_water

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get user input from form
        age = int(request.form['age'])
        gender = request.form['gender']
        bmi = float(request.form['bmi'])
        workout_frequency = int(request.form['workout_frequency'])
        experience_level = int(request.form['experience_level'])
        workout_type = request.form['workout_type']
        fat_percentage = float(request.form['fat_percentage'])
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        max_bpm = int(request.form['max_bpm'])
        avg_bpm = int(request.form['avg_bpm'])
        resting_bpm = int(request.form['resting_bpm'])
        session_duration = float(request.form['session_duration'])

        # Encoding gender and workout type
        gender_encoded = label_encoders['Gender'].transform([gender])[0]
        workout_type_encoded = label_encoders['Workout_Type'].transform([workout_type])[0]

        # Level 1 Prediction (Fitness Goal)
        level1_input = pd.DataFrame([[age, gender_encoded, bmi, workout_frequency,
                                      experience_level, workout_type_encoded, fat_percentage]],
                                    columns=level1_features)
        fitness_goal_pred = clf.predict(level1_input)[0]
        fitness_goal_label = label_encoders['Fitness_Goal'].inverse_transform([fitness_goal_pred])[0]

        # Level 2 Prediction (Calories Burned)
        level2_input = scaler.transform([[age, gender_encoded, weight, height, max_bpm, avg_bpm,
                                           resting_bpm, session_duration, workout_type_encoded,
                                           fat_percentage, bmi, fitness_goal_pred ]])
        calories_burned_pred = regressor.predict(level2_input)[0]

        # Level 3 Optimization (Workout Plan)
        optimal_plan = calculate_optimal_plan(user_experience=experience_level, max_bpm=max_bpm, 
                                              predicted_calories=calories_burned_pred, age=age, 
                                              fitness_goal=fitness_goal_label)

        # Extract optimal plan values
        session_duration_opt = optimal_plan[0]
        target_bpm = optimal_plan[1]
        fatigue_risk = optimal_plan[2]

        # Modify the Fatigue Risk based on its value and Workout Type based on Fatigue Risk
        if fatigue_risk < 0.3:
            fatigue_risk = "Low"
            optimal_workout_type = 'Strength'

        elif fatigue_risk < 0.7:
            fatigue_risk = "Medium"
            optimal_workout_type = 'Cardio'
        else:
            fatigue_risk = "High"
            optimal_workout_type = 'HIIT'

        

        # Calculate the recommended water intake
        recommended_water = round(calculate_water_intake(weight, session_duration_opt))

        return render_template('result.html', fitness_goal=fitness_goal_label, 
                               calories_burned=calories_burned_pred, optimal_plan=optimal_plan, 
                               optimal_workout_type=optimal_workout_type, recommended_water=recommended_water,
                               session_duration=session_duration_opt, target_bpm=target_bpm, 
                               fatigue_risk=fatigue_risk)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

