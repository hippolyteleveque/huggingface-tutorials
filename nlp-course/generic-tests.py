from transformers import pipeline


# Sentiment analysis
classifier = pipeline("sentiment-analysis")
res_sentiment_analysis = classifier(
    "My day was extremely deceptive as I did not have enough time to study machine learning."
)

# Zero shot classification
classifier = pipeline("zero-shot-classification")
res_zero_shot_clf = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)

# Text generation
generator = pipeline("text-generation")
generated_text = generator("In this course, we will teach you how to")

# Using specific model on the hub
generator = pipeline("text-generation", model="distilgpt2")
generated_text_distilgpt2 = generator(
    "In this course, we will teach you how to", max_length=30, num_return_sequences=2
)

# Mask filling
unmasker = pipeline("fill-mask")
res_mask_filling = unmasker(
    "This course will teach you all about <mask> models", top_k=2
)

# Named entity recognition
ner = pipeline("ner", grouped_entities=True)
res_ner = ner("My name is Hippolyte and I am a software engineer in Paris")

# Question answering
question_answerer = pipeline("question-answering")
res_qa = question_answerer(
    question="What is my job ?",
    context="My name is Hippolyte and I am a software engineer in Paris",
)

# Summarization
summarizer = pipeline("summarization")
summary = summarizer(
    """Animals die, friends die, and I shall die. One thing never dies, and that is the reputation we leave behind at our death.

So (apparently) said the Vikings, and, as it happens, they were bang on the money. Because, while it’s been over 900 years since the legendary Scandinavian warriors roamed our shores, we’re still every bit as fascinated with the (admittedly barbaric) raiders and their culture as ever – even if we can’t remember whether or not there’s a great-great-great-great-great-great uncle Olaf hidden somewhere in the branches of our family tree.

Which is why we were so excited to learn that there’s a very easy way to determine whether or not there’s any Viking blood coursing through our veins: take a closer look at our surnames.
Oh yes, a rose by any other name may smell as sweet – but a Scandinavian warrior is all about their moniker.

Experts have said that any surname ending in ‘sen’ or ‘son’ is likely to be of Viking descent (big news for Emma Watson, Emma Thompson, Robert Pattinson and co) – and surnames such as Roger/s, Rogerson, and Rendall also hint that there’s a touch of the marauder to you.

And they aren’t the only surnames that wannabe Vikings should watch out for."""
)


# Translation
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translation = translator("Ce cours est produit par Hugging Face.")

if __name__ == "__main__":
    print(f"Result for sentiment analysis task: {res_sentiment_analysis}")
    print(f"Result for zero shot classification: {res_zero_shot_clf}")
    print(f"Result for Text generation : {generated_text}")
    print(f"Result for text generation with distilgpt2: {generated_text_distilgpt2}")
    print(f"Result for named entity recognition: {res_ner}")
    print(f"Result for mask filling: {res_mask_filling}")
    print(f"Result fo question answering: {res_qa}")
    print(f"Result for summarization: {summary}")
    print(f"Result for translation: {translation}")
