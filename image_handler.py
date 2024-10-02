#from llama_cpp import Llama
#from llama_cpp.llama_chat_format import Llava15ChatHandler
import base64
import requests
from utils import load_config
#import streamlit as st
config = load_config()
import google.generativeai as genai
import os
import io
from PIL import Image, UnidentifiedImageError

def convert_bytes_to_base64(image_bytes):
    encoded_string=  base64.b64encode(image_bytes).decode("utf-8")
    return "data:image/jpeg;base64," + encoded_string

def upload_to_gemini(image_bytes, mime_type="image/jpeg"):
    """Uploads the given image bytes to Gemini after validating the image."""
    
    # Validate the image format
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image_format = image.format
        if image_format not in ["PNG", "JPEG"]:
            raise UnidentifiedImageError("Unsupported image format. Please upload a PNG or JPEG file.")
    except UnidentifiedImageError as e:
        raise ValueError(f"Error: {str(e)}")

    temp_image_path = "temp_image." + image_format.lower()
    with open(temp_image_path, 'wb') as f:
        f.write(image_bytes)

    file = genai.upload_file(temp_image_path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")

    # Optionally, clean up the temporary file
    os.remove(temp_image_path)

    return file

def load_vision():
    api_key = ""
    #config["gemini_api"]["api_key"]
    genai.configure(api_key=api_key)

    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    # safety_settings = Adjust safety settings
    # See https://ai.google.dev/gemini-api/docs/safety-settings
    system_instruction="You are an assistant who perfectly describes images.",
    )

    return model


#@st.cache_resource # can be cached if you use it often
#def load_llava():
#    chat_handler = Llava15ChatHandler(clip_model_path=config["llava_model"]["clip_model_path"])
#    llm = Llama(
#        model_path=config["llava_model"]["llava_model_path"],
#        chat_handler=chat_handler,
#        logits_all=True,
#        n_ctx=1024 # n_ctx should be increased to accomodate the image embedding
#       )
#    return llm


def handle_image(image_bytes, user_message):

    model = load_vision()

    try:
        # Upload the image bytes to Gemini
        uploaded_file = upload_to_gemini(image_bytes)

        # Start a chat session with the uploaded image and user message
        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        uploaded_file,
                        user_message,
                    ],
                }
            ]
        )

        # Send a message to the model and get the response
        response = chat_session.send_message(user_message)
        return response.text

    except ValueError as e:
        return str(e)

    #output = llava.create_chat_completion(
    #    messages = [
    #        {"role": "system", "content": "You are an assistant who perfectly describes images."},
    #        {
    #            "role": "user",
    #            "content": [
    #                {"type": "image_url", "image_url": {"url": image_base64}},
    #                {"type" : "text", "text": user_message}
    #            ]
    #        }
    #    ]
    #)
    #print(output)
    #return output["choices"][0]["message"]["content"]