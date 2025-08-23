from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from transformers import pipeline


# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
# model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

# inputs = tokenizer("Hello world!", return_tensors="pt")
# outputs = model(**inputs)


classifier = pipeline("sentiment-analysis", cache_dir="./model")
output = classifier(
    "We are very happy to introduce pipeline to the transformers repository."
)
print(output)
