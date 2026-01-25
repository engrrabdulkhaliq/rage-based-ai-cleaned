# import os

# structure = {
#     "web": {
#         "templates": {
#             "index.html": ""
#         },
#         "static": {
#             "style.css": "",
#             "chat.js": ""
#         },
#         "server.py": ""
#     }
# }

# def create_structure(base_path, tree):
#     for name, content in tree.items():
#         path = os.path.join(base_path, name)

#         if isinstance(content, dict):
#             os.makedirs(path, exist_ok=True)
#             create_structure(path, content)
#         else:
#             with open(path, "w", encoding="utf-8") as f:
#                 f.write(content)

# create_structure(".", structure)

# print("✅ Web UI folder structure created successfully")
# from openai import OpenAI


# client = OpenAI(api_key="AIzaSyDBTB1DDHMvT2ZlWaGlsPJlsaZlsGXIulk")

# response = client.responses.create(
#     model="gpt-4.1-mini",
#     input="Hello, are you working?"
# )

# print(response.output_text)

import google.generativeai as genai

# 🔑 Your API key
genai.configure(api_key="AIzaSyDBTB1DDHMvT2ZlWaGlsPJlsaZlsGXIulk")

# Choose the model
model = genai.GenerativeModel("gemini-3-flash-preview")

# Generate text
response = model.generate_content(
    "Explain Python decorators in simple words"
)

print(response.text)
