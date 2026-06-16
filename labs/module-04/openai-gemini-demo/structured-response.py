from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Ensure the API key is set in the environment
BASE_URL = os.getenv("BASE_URL")  # Ensure the base URL is set in the environment
MODEL_NAME = os.getenv("MODEL_NAME")  # Ensure the model name is set in the environment


client = OpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)

class ProductReview(BaseModel):
    product_name: str
    rating: float
    pros: list[str]
    cons: list[str]
    recommended: bool
    review_date: str | None

response = client.chat.completions.parse(
    model=MODEL_NAME,
    messages=[{"role": "system", "content": "Extract structured data."},
              {"role": "user", "content": "Review: I recently bought the XYZ headphones and I'm quite impressed. The sound quality is excellent, and they are very comfortable to wear for long periods. However, the battery life is a bit disappointing, lasting only about 4 hours on a full charge. Overall, I would recommend these headphones to anyone looking for great sound and comfort, but be prepared for the short battery life. Review date: 2024-06-01."}],
    response_format=ProductReview
)

print(f"Full Response: {response}")

print(response.choices[0].message.content)

# Token usage (important for cost tracking)
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")