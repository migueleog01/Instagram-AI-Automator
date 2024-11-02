from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Initialize the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_image_description(image_path, max_new_tokens=60, num_beams=5, top_k=50, top_p=0.9, do_sample=True):
    """
    Generates a detailed description for the given image.
    
    Parameters:
    - image_path (str): Path to the image file.
    - max_new_tokens (int): Maximum length for the generated description.
    - num_beams (int): Number of beams for beam search.
    - top_k (int): Limits the sampling to top k tokens.
    - top_p (float): Nucleus sampling threshold.
    - do_sample (bool): Enables sampling for diverse outputs.

    Returns:
    - str: Generated detailed description of the image content.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    
    # Process the image for the BLIP model
    inputs = processor(images=image, return_tensors="pt")
    
    # Generate the caption with fine-tuned parameters
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample
    )
    
    # Decode and return the description
    description = processor.decode(outputs[0], skip_special_tokens=True)
    return description

# Test with an example image
print(generate_image_description("../images/input/dog_playing_fetch.jpg"))
# print(generate_image_description("../images/input/sunrise.jpg"))
# print(generate_image_description("../images/input/pumpkin.jpeg"))
# print(generate_image_description("../images/input/basketball.jpeg"))
# print(generate_image_description("../images/input/plane.jpeg"))
#print(generate_image_description("../images/input/messi.webp"))
