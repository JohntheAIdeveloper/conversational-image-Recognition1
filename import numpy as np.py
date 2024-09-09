import numpy as np
import time
import os
import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import AutoModelForCausalLM, AutoTokenizer
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
import cv2  # For camera capture

# Download Microsoft's DialoGPT model and tokenizer for text conversations
checkpoint = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

# Load pre-trained MobileNetV2 model for image recognition
image_model = MobileNetV2(weights='imagenet')

# A ChatBot class with image recognition capabilities
class ChatBot():
    def __init__(self):
        self.chat_history_ids = None
        self.bot_input_ids = None
        self.end_chat = False
        self.welcome()
        
    def welcome(self):
        print("Initializing ChatBot ...")
        time.sleep(2)
        print('Type "bye" or "quit" or "exit" to end chat \n')
        print('Type "camera" to take a photo and recognize an image\n')
        time.sleep(3)
        greeting = np.random.choice([
            "Welcome, I am ChatBot, here for your kind service",
            "Hey, Great day! I am your virtual assistant",
            "Hello, it's my pleasure meeting you",
            "Hi, I am a ChatBot. Let's chat!"
        ])
        print("ChatBot >>  " + greeting)
        
    def user_input(self):
        text = input("User    >> ")
        if text.lower().strip() in ['bye', 'quit', 'exit']:
            self.end_chat = True
            print('ChatBot >>  See you soon! Bye!')
            time.sleep(1)
            print('\nQuitting ChatBot ...')
        elif text.lower().startswith("http"):  # Check if input is an image URL
            try:
                self.image_recognition_url(text)
            except Exception as e:
                print(f"ChatBot >>  Failed to process the image. Error: {e}")
        elif text.lower().strip() == 'camera':  # Use camera to capture image
            try:
                self.image_recognition_camera()
            except Exception as e:
                print(f"ChatBot >>  Failed to process the camera image. Error: {e}")
        else:
            # encode the new user input, add the eos_token and return a tensor in Pytorch
            self.new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')
    
    def image_recognition_url(self, image_url):
        print("ChatBot >>  Processing the image from URL...")
        
        # Download image from URL
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        
        # Preprocess the image for MobileNetV2
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Perform image recognition
        preds = image_model.predict(img_array)
        decoded_preds = decode_predictions(preds, top=3)[0]
        
        # Create a response based on top predictions
        response_text = "I think this is a {} with a {:.2f}% confidence.".format(
            decoded_preds[0][1], decoded_preds[0][2] * 100
        )
        print(f"ChatBot >>  {response_text}")
    
    def image_recognition_camera(self):
        print("ChatBot >>  Opening the camera... Press 'q' to capture.")
        
        # Start video capture
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ChatBot >>  Failed to capture image.")
                break
                
            # Display the captured frame
            cv2.imshow('Press "q" to capture', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # Save the captured frame as an image
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = img.resize((224, 224))
                
                # Preprocess the image for MobileNetV2
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                
                # Perform image recognition
                preds = image_model.predict(img_array)
                decoded_preds = decode_predictions(preds, top=3)[0]
                
                # Create a response based on top predictions
                response_text = "I think this is a {} with a {:.2f}% confidence.".format(
                    decoded_preds[0][1], decoded_preds[0][2] * 100
                )
                print(f"ChatBot >>  {response_text}")
                break
        
        # Release the camera and close any open windows
        cap.release()
        cv2.destroyAllWindows()
    
    def bot_response(self):
        if self.chat_history_ids is not None:
            self.bot_input_ids = torch.cat([self.chat_history_ids, self.new_user_input_ids], dim=-1)
        else:
            self.bot_input_ids = self.new_user_input_ids

        self.chat_history_ids = model.generate(self.bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        
        if response == "":
            response = self.random_response()
        print('ChatBot >>  ' + response)
        
    def random_response(self):
        i = -1
        response = tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[i]:][0], skip_special_tokens=True)
        while response == '':
            i = i - 1
            response = tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[i]:][0], skip_special_tokens=True)
        
        if response.strip() == '?':
            reply = np.random.choice(["I don't know", "I am not sure"])
        else:
            reply = np.random.choice(["Great", "Fine. What's up?", "Okay"])
        return reply

# Build a ChatBot object and start chatting
bot = ChatBot()
while True:
    bot.user_input()
    if bot.end_chat:
        break
    if bot.bot_input_ids is not None:
        bot.bot_response()
