from typing import List

from anthropic import Anthropic
from PIL import Image

from .embedder import ImageProcessor


class RAGProcessor:
    def __init__(self):
        # Anthropic() picks up ANTHROPIC_API_KEY from the environment (loaded via load_dotenv()).
        self.client = Anthropic()

    def generate_response(self, user_prompt: str, user_image: Image.Image,
                          context_images: List[Image.Image]) -> str:
        context_image_base64_list = [ImageProcessor.image_to_base64(img) for img in context_images]

        if user_image is not None:
            user_image_base64 = ImageProcessor.image_to_base64(user_image)
            prompt_with_context = (
                f"User Prompt: {user_prompt}\n\n"
                f"Instructions: Use the 'User Image' as the primary reference for answering the query. "
                f"Use the 'Database Images' as contexts to inform and help you formulate the response. "
                f"Below your main response, the 'Database Images' will be displayed to the user. So always be aware of their existence with sentence like 'as you can see in the images below' or any other variations, depending on the user's query. "
                f"The images are provided below, with the 'User Image' first, followed by {len(context_images)} 'Database Images'."
            )
            content = [
                {"type": "text", "text": prompt_with_context},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": user_image_base64}},
                *[
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_base64}}
                    for img_base64 in context_image_base64_list
                ],
            ]
        else:
            prompt_with_context = (
                f"User Prompt: {user_prompt}\n\n"
                f"Instructions: Use the 'Database Images' provided below to inform and help you formulate the response to the user's query. "
                f"All images shown are from the database and represent similar or relevant content based on the user's text prompt. "
                f"Below your main response, these 'Database Images' will be displayed to the user. So always be aware of their existence with sentence like 'as you can see in the images below' or any other variations, depending on the user's query. "
                f"There are {len(context_images)} 'Database Images' provided below."
            )
            content = [
                {"type": "text", "text": prompt_with_context},
                *[
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_base64}}
                    for img_base64 in context_image_base64_list
                ],
            ]

        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": content}],
        )
        return response.content[0].text
