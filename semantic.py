import spacy
nlp = spacy.load('en_core_web_sm')

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

tokens = nlp('cat apple monkey banana')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

sentence_to_compare = "why is my cat on the car"

sentences = ["Where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"
]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

#================Observations
#Cat is more similar to monkey than monkey is to banana, as they are both animals
#Banana is not at all similar to cat
#Cat is itself so scores 1.0 similarity. Same for Monkey, Apple and Banana
#Cat and monkey gets >.5 similarity because they are both in the same class (animals)
#Apple and banana are very similar as they are in the class of fruit, but they are not very similar to cat and monkey
#Which are similar to each other as they're in the same class of animal
#The observation pointing out the car has the most similarity to the test string. For whatever reason the entry
#"I've lost my car in my car" is less similar, presumably because it is not directly implying an observation of the car.
#When swapped to the sm model a UserWarning is generated, illustrating the difficulties that model may have.