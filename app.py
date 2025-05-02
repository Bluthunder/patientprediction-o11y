import gradio as gr
import joblib
import numpy as np

# Load the trained model
trained_model = joblib.load('xgboost-model_v1_opt.pkl')

def predict_death(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time):

    # Prepare the input data for the model
    input_data = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time]])

    # Make predictions
    prediction = trained_model.predict(input_data)[0]
    probability = trained_model.predict_proba(input_data)[0][1]

    # Return result as a message
    result = "🟥 Patient is likely to die" if prediction == 1 else "🟩 Patient is likely to survive"
    return f"{result}\nRisk Score: {probability:.2f}"


# Define input components
inputs = [
    gr.Slider(20, 100, step=1, label="Age"),
    gr.Radio([0, 1], label="Anaemia (0 = No, 1 = Yes)"),
    gr.Slider(20, 800, step=1, label="Creatinine Phosphokinase"),
    gr.Radio([0, 1], label="Diabetes (0 = No, 1 = Yes)"),
    gr.Slider(10, 80, step=1, label="Ejection Fraction (%)"),
    gr.Radio([0, 1], label="High Blood Pressure (0 = No, 1 = Yes)"),
    gr.Slider(50000, 600000, step=1000, label="Platelets"),
    gr.Slider(0.1, 10.0, step=0.1, label="Serum Creatinine"),
    gr.Slider(100, 150, step=1, label="Serum Sodium"),
    gr.Radio([0, 1], label="Sex (0 = Female, 1 = Male)"),
    gr.Radio([0, 1], label="Smoking (0 = No, 1 = Yes)"),
    gr.Slider(1, 10, step=1, label='Time')
]

# Define output component
outputs = gr.Textbox(label="Prediction Result", lines=2)

title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

iface = gr.Interface(fn=predict_death,
                     inputs=inputs,
                     outputs=outputs,
                     title=title,
                     description=description,
                     allow_flagging='never')

iface.launch(server_name="0.0.0.0", server_port=7860)

