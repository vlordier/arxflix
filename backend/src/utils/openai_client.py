""" Description: OpenAI client for generating video scripts. """

import openai

from backend.src.config import OPENAI_API_KEY


class OpenAIClient:
    def __init__(self, api_key: str = OPENAI_API_KEY):
        """Initialize the OpenAI client.

        Args:
            api_key (str): The OpenAI API key.

        """
        self.client = openai.OpenAI(api_key=api_key)

    def generate_script(self, model: str, system_prompt: str, user_content: str) -> str:
        """
        Generate a video script using OpenAI's GPT-4o model.

        Args:
            model (str): The model to use for generating the script.
            system_prompt (str): The system prompt for the script generation.
            user_content (str): The user content to generate the script from.

        """
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        result = response.choices[0].message.content

        if not result:
            raise ValueError("No result returned from OpenAI.")

        return result
