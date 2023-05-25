# Create a file called semantic.py and run all the code extracts above.

import spacy
nlp = spacy.load('en_core_web_md')
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

#Write a note about what you found interesting about the similarities
#between cat, monkey and banana and think of an example of your own.

''' I find it interesting that the program manages to indentify that monkey is connected more to banana than the cat is; 
And that it finds monkey and cat to be the most similar words, as they are both animals.
It is truly fascinating to think the kind of learning and programming took place to achieve this.
'''
# my example:

nlp = spacy.load('en_core_web_md')
word1 = nlp("water")
word2 = nlp("fish")
word3 = nlp("cat")
print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

'''
I wonder why the program finds much less similarity between cat and water, with almost no similarity (>0.1),
as cats are famously connected with water (in a negative way; they hate it).
I'm not surpirsed that the fish was found to be the most similar to water,
but I was expecting a higher similarity. Cat and fish are more related than cat and water but I would expect a higher similarity.
'''

sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]
model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)



# Run the example file with the simpler language model ‘en_core_web_sm’
# and write a note on what you notice is different from the model
# 'en_core_web_md'.

'''
When I run it I get the following warning/message:
UserWarning: [W007] The model you're using has no word vectors loaded, so the result of the Doc.similarity method will be 
based on the tagger, parser and NER, which may not give useful similarity judgements. This may happen if you're using one 
of the small models, e.g. `en_core_web_sm`, which don't ship with word vectors and only use context-sensitive tensors. 
You can always add your own word vectors, or use one of the larger models instead if available.
'''