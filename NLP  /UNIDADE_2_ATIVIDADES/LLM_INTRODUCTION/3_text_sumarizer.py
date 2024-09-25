from transformers import pipeline

llm = pipeline("summarization", model="facebook/bart-large-cnn")
long_text = "Life requires care. Love too. Flowers and thorns are beauties that go together. Don't want just one, they don't know how to live alone... Whoever wants to take the rose into their life will have to know that with them come countless thorns. Don't worry, the beauty of the rose is worth the hassle of the thorns."

outputs = llm(long_text, max_length=60, clean_up_tokenization_spaces=True)
print(outputs[0]['summary_text'])