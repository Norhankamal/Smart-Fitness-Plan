# 🧠 Smart Fitness Plan – ML-based Health & Exercise Recommendation System

## 📋 Description
This project is an **AI-powered fitness assistant** that helps users improve their health through **Machine Learning predictions**.  
It performs three main tasks:  
1. **Classifies the user’s goal** (e.g. weight loss, muscle gain, or fitness maintenance).  
2. **Predicts the calories** the user should burn based on their data.  
3. **Generates a customized fitness plan** that includes:  
   - Workout types and duration  
   - Water intake recommendation  
   - Training hours per day  

All three models are connected — each step’s output is used as an input for the next one, ensuring personalized and accurate results.  
Later, the project was deployed using **Flask** to create an interactive web application.

---

## ⚙️ Tech Stack
- **Python 3.9+**
- **Flask**
- **Scikit-learn**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**

---

## 🚀 Features
✅ Goal classification using supervised ML  
✅ Calorie burn prediction  
✅ Smart plan recommendation  
✅ Flask web interface for user interaction  

---

## 🧩 Project Structure
```
├── functionsHM.ipynb        # Core ML functions (classification, calorie prediction, plan generation)
├── finalHMbeforeflask.ipynb # Model training and testing before Flask integration
├── app.py                   # Flask web app (for deployment)
├── static/ & templates/     # Web UI components
```

---

## ⚡ How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/YourUsername/Smart-Fitness-Plan.git
cd Smart-Fitness-Plan
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Flask app
```bash
python app.py
```

### 4️⃣ Open in browser
Go to 👉 `http://127.0.0.1:5000`

---

## 💡 Future Enhancements
- Connect wearable device data for real-time tracking
- Add chatbot or voice assistant integration
- Improve UI/UX with dashboards and visual progress tracking

---

## 🧑‍💻 Author
**Norahan Alla**  
AI & ML Developer | Passionate about health tech and intelligent systems
