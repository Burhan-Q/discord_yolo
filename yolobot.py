"""
Burhan Qaddoumi
2023-04-25
Simple Discord YOLO bot
"""
import discord
import torch
import torchvision
import urllib.request
from PIL import Image
from io import BytesIO

# Discord client
client = discord.Client()

# Load the pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Object detection via URL image
def detect_objects(image_url):
    # Download the image from the URL
    response = urllib.request.urlopen(image_url)
    image_data = response.read()
    
    # Open the image using PIL
    image = Image.open(BytesIO(image_data)) # NOTE may need to resize image for model
    
    # Model inference
    results = model(image)
    
    # Extract the class labels and confidence scores for each detection
    labels = results.xyxyn[0][:, -1].numpy()
    scores = results.xyxyn[0][:, -2].numpy()
    
    # Construct list of strings describing each detection
    objects = []
    for i in range(len(labels)):
        label = labels[i]
        score = scores[i]
        objects.append(f"{label}: {score:.2f}")
    
    # Return the list of detections
    return objects

@client.event
async def on_message(message):
  """Detection response to incoming Discord messages with image attachments"""
    # Check if message contains an image attachment
    if len(message.attachments) > 0 and message.attachments[0].url.endswith(('png', 'jpg', 'jpeg')):
        # Get URL to image
        image_url = message.attachments[0].url
        
        # Run inference on image
        objects = detect_objects(image_url)
        
        # Reply with detected objects
        response = "Detected objects:\n"
        response += "\n".join(objects)
        await message.channel.send(response)

# Run the Discord bot
client.run('your_discord_bot_token_here') # TODO add method for importing token
