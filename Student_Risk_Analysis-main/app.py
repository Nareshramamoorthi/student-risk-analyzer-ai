import gradio as gr
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from gtts import gTTS
import os

# Load dataset and preprocess
data = pd.read_csv("Student_Performance.csv")

# Encode categorical data
encoder = LabelEncoder()
data["Extracurricular Activities"] = encoder.fit_transform(data["Extracurricular Activities"])

# Prepare data for training
Train = data.drop(columns="Performance Index")
Target = data["Performance Index"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(Train, Target, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Function to analyze improvements
def analyze_improvement(hours_studied, previous_scores, extracurricular_activities, sleep_hours, sample_question_papers_practiced):
    feedback = []

    if hours_studied < 3:
        feedback.append("Increase your study hours to at least 3 hours a day to reinforce learning.")
    if previous_scores < 70:
        feedback.append("Review and strengthen foundational concepts, especially areas where you struggled.")
    if extracurricular_activities == 0:
        feedback.append("Engage in extracurricular activities to develop teamwork, time management, and stress relief.")
    if sleep_hours < 7:
        feedback.append("Aim for at least 7 hours of sleep per night to improve cognitive function.")
    if sample_question_papers_practiced < 5:
        feedback.append("Increase practice with sample question papers to improve test-taking strategies.")
    
    return feedback

# Function to predict performance and generate feedback
def predict_and_generate_feedback(hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers):
    # Convert 'Yes'/'No' extracurricular response to numerical
    extracurricular = 1 if (extracurricular and extracurricular.lower() == "yes") else 0

    # Create input array for the model
    user_input = np.array([[hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers]])

    # Predict performance
    predicted_performance = np.round(model.predict(user_input), decimals=1)[0]
    result = "Pass" if predicted_performance >= 50 else "Fail"

    # Generate improvement feedback
    feedback = analyze_improvement(hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers)
    feedback_text = "\n".join(feedback) if feedback else "Keep up the good work!"

    # Convert feedback to speech and save it as an audio file
    tts = gTTS(text=feedback_text, lang='en')
    audio_path = "feedback_audio.mp3"
    tts.save(audio_path)

    return predicted_performance, result, feedback_text, audio_path

# Function to generate PDF report
def generate_pdf(hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers, predicted_performance):
    extracurricular = 1 if extracurricular.lower() == "yes" else 0
    feedback = analyze_improvement(hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers)

    pdf_path = "student_improvement_report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.drawString(100, 750, "Student Performance Improvement Report")
    c.drawString(100, 730, f"Predicted Performance: {predicted_performance}")
    c.drawString(100, 710, "Suggested Improvements:")

    y_position = 690
    for i, tip in enumerate(feedback, 1):
        c.drawString(100, y_position, f"{i}. {tip}")
        y_position -= 20

    c.save()
    return pdf_path

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## Student Performance Prediction")
    
    # Dark and Light mode toggle
    mode = gr.Checkbox(label="Dark Mode", value=False)
    
    # Input fields
    with gr.Row():
        hours_studied = gr.Number(label="Hours Studied")
        previous_scores = gr.Number(label="Previous Scores")
        extracurricular = gr.Radio(["Yes", "No"], label="Extracurricular Activities")
        sleep_hours = gr.Number(label="Sleep Hours")
        sample_papers = gr.Number(label="Sample Question Papers Practiced")
    
    # Prediction and feedback
    predict_button = gr.Button("Predict and Generate Feedback")
    prediction_output = gr.Textbox(label="Predicted Performance")
    result_output = gr.Textbox(label="Result (Pass/Fail)")
    feedback_output = gr.Textbox(label="Improvement Suggestions")
    audio_output = gr.Audio(label="Audio Feedback")

    # Generate PDF report
    pdf_button = gr.Button("Download PDF Report")
    pdf_output = gr.File(label="Download Report")
    
    with gr.Row():
       
        image2 = gr.Image("images/img2.png", label="Performance by Sleeping Hours")
        image7 = gr.Image("images/img6.png", label="Performance by Hours Studied")
        image8 = gr.Image("images/img7.png", label="Performance by Extracurricular Activities")
        
    with gr.Row():
       
        image4 = gr.Image("images/img3.png", label="Extracurricular Activities Pie chart")
        image7 = gr.Image("images/img9.png", label="Heat Map")
        image9 = gr.Image("images/img10.png", label="Linear Regression")


    # Function to toggle dark mode
    def toggle_mode(is_dark):
        if is_dark:
            return "dark"
        else:
            return "default"

    mode.change(fn=toggle_mode, inputs=mode, outputs=demo)

    # Predict and feedback function call
    predict_button.click(
        fn=predict_and_generate_feedback,
        inputs=[hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers],
        outputs=[prediction_output, result_output, feedback_output, audio_output]
    )

    # Generate PDF report function call
    pdf_button.click(
        fn=generate_pdf,
        inputs=[hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers, prediction_output],
        outputs=pdf_output
    )

demo.launch()
