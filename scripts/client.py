import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# create `.env` and add OPENAI_API_KEY=your_api_key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://chat.intern-ai.org.cn/api/v1/",
)

chat_rsp = client.chat.completions.create(
    model="intern-latest",
    # images and texts in one single
    messages=[
        {"role": "user", "content": "hello"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image please"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://static.openxlab.org.cn/internvl/demo/visionpro.png"  # image by url
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,{<encode_image(image_path)>}"  # image by base64: replace <encode_image(image_path)> with base64 of image
                    },
                },
            ],
        },
    ],  # role: user, system, assistant
    temperature=0.3,
    extra_body={
        "enable_thinking": False,
    },
    stream=True,
    # function_tools, top_p, max_tokens, etc.
)

# # Stream=True
for chunk in chat_rsp:
    print(chunk.choices[0].delta.content)

# # Stream=False
# for choice in chat_rsp.choices:
#     print(choice.message.content)
