import cv2
import base64
import numpy as np
import openai
import requests
import os

# Set your OpenAI API key
openai.api_key = ''

# Load the video
video = cv2.VideoCapture("bjj1.mp4")

# Calculate video length and read frames, encoding them to base64
base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

# Release the video object
video.release()
print(len(base64Frames), "frames read.")

# Create a directory to save frames
os.makedirs("frames", exist_ok=True)

# Save each frame as a JPEG file
for i, img in enumerate(base64Frames):
    decoded_img = base64.b64decode(img.encode("utf-8"))
    np_img = np.frombuffer(decoded_img, dtype=np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    frame_filename = f"frames/frame_{i}.jpg"
    cv2.imwrite(frame_filename, frame)

print("Frames saved.")

# Create OpenAI chat completion
PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            "These are frames of a video. Create a short voiceover script in the style of a super excited Brazilian sports narrator who is narrating his favorite match. He is a big fan of JiuJitsu. Use caps and exclamation marks where needed to communicate excitement. Only include the narration, your output must be in English. When the fighter submit the opponent, you must scream THAT'S A TAP either once or multiple times.",
            *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::10]),
        ],
    },
]

result = openai.ChatCompletion.create(
    model="gpt-4-vision-preview",
    messages=PROMPT_MESSAGES,
    max_tokens=500
)
print(result.choices[0].message.content)

# Post request to TTS API
response = requests.post(
    "https://api.openai.com/v1/audio/speech",
    headers={
        "Authorization": f"Bearer {openai.api_key}",
    },
    json={
        "model": "tts-1",
        "input": result.choices[0].message.content,
        "voice": "fable",
    },
)

# Save the audio response as an MP3 file
audio = b""
for chunk in response.iter_content(chunk_size=1024 * 1024):
    audio += chunk

with open('output.mp3', 'wb') as file:
    file.write(audio)

print("The MP3 file has been saved locally as 'output.mp3'.")