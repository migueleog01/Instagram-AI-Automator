import openai

#get key from env
import os
from dotenv import load_dotenv

# Set your OpenAI API key
openai.api_key =  os.getenv("OPENAI_API_KEY")

def generate_caption_and_hashtags(description):
    """
    Generates a caption and hashtags based on an image description using OpenAI's GPT-4.

    Parameters:
    - description (str): The description of the image content.

    Returns:
    - dict: A dictionary with generated 'caption' and 'hashtags'.
    """
    prompt = (
        f"Create a catchy Instagram caption and 5-10 relevant hashtags based on this image description:\n\n"
        f"Description: {description}\n\n"
        "Caption:\n\n"
        "Hashtags:"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an assistant that helps generate social media captions and hashtags."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.7
    )

    # Extract the text from the response
    text = response['choices'][0]['message']['content'].strip()

    # Separate caption and hashtags
    if "Caption:" in text and "Hashtags:" in text:
        caption_part = text.split("Caption:")[1].split("Hashtags:")[0].strip()
        hashtags_part = text.split("Hashtags:")[1].strip()
        hashtags_list = hashtags_part.split()  # Split hashtags by whitespace

        return {
            "caption": caption_part,
            "hashtags": hashtags_list
        }

    return {
        "caption": "No caption generated.",
        "hashtags": []
    }

# Example usage
if __name__ == "__main__":
    description = "A dog running with a ball in its mouth"
    result = generate_caption_and_hashtags(description)
    print("Caption:", result["caption"])
    print("Hashtags:", result["hashtags"])
