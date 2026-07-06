# 📧 Spam Email Detection using Machine Learning

## 📌 Project Overview
This project is a **Spam Email Classifier** built using Machine Learning.  
It detects whether an email message is **Spam** or **Ham (Not Spam)** using Natural Language Processing techniques.

The model is trained using a dataset of emails stored in folders (`spam` and `ham`).

---

## ⚙️ Technologies Used
- Python 🐍
- Pandas
- NumPy
- Scikit-learn
  - CountVectorizer
  - Multinomial Naive Bayes

---

## 📂 Dataset Structure
The dataset is organized as follows:
emails/
├── ham/ → Normal emails
└── spam/ → Spam emails


- Total emails: ~3000  
- Spam emails: ~500  
- Ham emails: ~2500  

---

## 🧠 Machine Learning Workflow

1. Read emails from folders
2. Extract message content
3. Convert text into numerical features using **CountVectorizer**
4. Train model using **Multinomial Naive Bayes**
5. Predict whether new messages are Spam or Ham

---

## 🚀 How to Run the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
2. Run the program
python spam_detector.py
📊 Model Performance
Accuracy: ~98% (varies based on dataset split)
Works well for basic spam detection tasks
✉️ Example Predictions
Message	Prediction
Free Viagra now! Click to win money	Spam
Hi, are we meeting tomorrow?	Ham
Congratulations! You won a lottery prize	Spam

💡 Features

Reads real email dataset from folders
Text preprocessing using Bag of Words model
Fast and simple Naive Bayes classifier
Real-time message prediction support

👨‍💻 Author

Keshav (Student Project)

📌 Future Improvements
Use TF-IDF instead of CountVectorizer
Build a web app using Streamlit
Add deep learning model for better accuracy
Deploy on cloud (AWS / Heroku)