import google.generativeai as genai

genai.configure(api_key="AIzaSyDATrnMbi2SpQrJBTRDN77v-sQX1HkA-ks") 

print("Available Gemini models that support generateContent:")
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(model.name)

print("\nAvailable Gemini models that support embedContent:")
for model in genai.list_models():
    if 'embedContent' in model.supported_generation_methods:
        print(model.name)