from transformers import pipeline

llm = pipeline("text-generation")
prompt = "The color pink is beautiful because"
outputs = llm(prompt, max_length=25)

print(outputs[0]['generated_text'])