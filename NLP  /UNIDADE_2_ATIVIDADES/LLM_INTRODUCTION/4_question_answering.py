from transformers import pipeline

llm = pipeline("question-answering")
context = "Cats more affectionate than dogs"

question = "Why are cats more affectionate than dogs?"
outputs = llm(question=question, context=context)
print(outputs['answer'])