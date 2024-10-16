import gradio as gr
import os
from groq import Groq
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tensorflow.keras.models import load_model
import gradio as gr
import os
from groq import Groq

import requests
from keras.models import load_model

# Download the file from Google Drive (use the direct download link, not the view link)
url = "https://drive.google.com/uc?export=download&id=1a7XENBfSYjuoPnwEn_asnhg1lt4qoRRV"
response = requests.get(url)

# Save the file locally
with open("CICI_Project(1).h5", "wb") as file:
    file.write(response.content)

# Load the model from the local file
model = load_model("CICI_Project(1).h5")




# Step 2: Preprocess the image before feeding it into the model
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),  # Resize the image to 150x150 pixels
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
    ])
    image_tensor = transform(image)


    # Reorder dimensions to match TensorFlow/Keras input format (Height, Width, Channels)
    image_tensor = image_tensor.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)
    image_tensor = np.expand_dims(image_tensor, axis=0)  # Add batch dimension


    return image_tensor


# Step 3: Define the function to detect tumor and interpret the result
def detect_tumor(image_array):
    # Make prediction
    predictions = model.predict(image_array)
    probability = predictions[0]  # Get the probabilities for the classes
    return probability


def interpret_result(probability):
    # Update the output to reflect CICI detection
    if probability[0] > 0.5:  # Assuming index 0 is 'No Tumor' and index 1 is 'Tumor'
        return f"Probability of Cognitive Impairment after Chemotherapy: {probability[0]*100:.2f}%"
    else:
        return f"Probability of Cognitive Impairment after Chemotherapy: {probability[1]*100:.2f}%"


# Step 4: Define treatment plans based on CICI detection probabilities
treatment_plans = {
    "low_risk": {
        "exercises": [
            "Daily aerobic exercises like walking or cycling",
            "Mind puzzles (crosswords, Sudoku, etc.)",
            "Reading books, especially those focusing on memory retention",
            "Social interaction activities (engage in group conversations)"
        ],
        "duration": "20-30 minutes per day",
        "description": "Low-intensity cognitive exercises focusing on mental stimulation and maintenance of cognitive function."
    },
    "moderate_risk": {
        "exercises": [
            "Memory enhancement games (e.g., card games focusing on short-term memory)",
            "Structured meditation for mindfulness and stress reduction",
            "Cognitive behavioral therapy (CBT) with a professional",
            "Weekly group therapy sessions focusing on cognitive improvement"
        ],
        "duration": "30-45 minutes per day",
        "description": "Moderate-intensity cognitive training exercises, targeting specific cognitive skills like memory retention, attention, and stress management."
    },
    "high_risk": {
        "exercises": [
            "Intensive cognitive rehabilitation sessions (computer-based cognitive training)",
            "Daily journaling of activities and thoughts to enhance memory recall",
            "Professional cognitive therapy (1-on-1 sessions)",
            "Interactive neuroplasticity exercises with a therapist (brain stimulation therapies)"
        ],
        "duration": "1-1.5 hours per day",
        "description": "High-intensity personalized rehabilitation plan focusing on cognitive recovery and brain function enhancement."
    }
}


# Step 5: Generate a treatment plan based on CICI probability
def generate_treatment_plan(cici_probability):
    if cici_probability <= 40:
        return treatment_plans["low_risk"]
    elif cici_probability <= 70:
        return treatment_plans["moderate_risk"]
    else:
        return treatment_plans["high_risk"]


# Step 6: Define the Gradio interface function to predict CICI and provide treatment
def predict_cici_and_treatment(image, patient_history):
    image_array = preprocess_image(image)
    probability = detect_tumor(image_array)


    # Interpret the result for CICI detection
    result = interpret_result(probability)


    # Extract CICI probability for treatment generation
    cici_probability = probability[1] * 100  # Assuming probability[1] is the correct one for 'Tumor' or impairment


    # Generate a personalized treatment plan based on CICI probability
    treatment_plan = generate_treatment_plan(cici_probability)


    # Format the treatment plan result
    treatment_result = f"Patient History: {patient_history}\nCICI Probability: {cici_probability:.2f}%\n\nPersonalized Treatment Plan:\n"
    treatment_result += f"- Exercises: {', '.join(treatment_plan['exercises'])}\n"
    treatment_result += f"- Duration: {treatment_plan['duration']}\n"
    treatment_result += f"- Description: {treatment_plan['description']}\n"


    return result + "\n\n" + treatment_result


# Step 7: Set up the Chatbot (Groq) integration
client = Groq(
    api_key=("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")  # Use environment variable for API key
)


def chat(message, history):
    try:
        # Prepare the messages including the history
        messages = [
            {"role": "system", "content": "You are a helpful healthcare assistant. Provide accurate and helpful information, but always advise consulting with a healthcare professional for medical advice."},
        ]
        for human, assistant in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": message})


        # Call the Groq API
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",
            max_tokens=1000,
            temperature=0.7,
        )


        # Extract and return the assistant's response
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"


# Step 8: Define the main Gradio interface with both the CICI detection and chatbot
with gr.Blocks() as interface:
    # Title
    gr.Markdown("<h1>Cognitive Impairment after Chemotherapy Detection & Personalized Treatment</h1>")


    # Row for CICI detection and treatment
    with gr.Row():
        with gr.Column():
            patient_history_input = gr.Textbox(label="Patient History (e.g., age, treatment status, symptoms)", lines=4)
            image_input = gr.Image(type="pil", label="Upload MRI Image")


        with gr.Column():
            result_output = gr.TextArea(label="Detection and Treatment Plan", lines=10)


    submit_button = gr.Button("Detect CICI and Get Treatment Plan")


    # Connect the predict function to button click
    submit_button.click(predict_cici_and_treatment, inputs=[image_input, patient_history_input], outputs=result_output)


    # Popup-like Chatbot for patient assistance
    with gr.Accordion("Chat with Healthcare Assistant", open=False):
        chatbot = gr.ChatInterface(
            fn=chat,
            title="Healthcare Assistant",
            description="Ask me any healthcare-related questions! Always consult with a healthcare professional.",
            theme="compact"
        )


# Launch the interface
interface.launch(share=True)
