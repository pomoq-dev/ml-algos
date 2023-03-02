import openai
import os

# Set your OpenAI API key
openai.api_key = "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"

# Define your prompt
prompt = "The quick brown fox jumps over the"

# Call the GPT-3 API to generate text based on your prompt
response = openai.Completion.create(
    engine="curie", prompt=prompt, max_tokens=100
)

# Print the generated text
print(response.choices[0].text)