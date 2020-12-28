import spacy
from scipy import spatial
import sys
import nltk
from nltk.stem import WordNetLemmatizer 
import re, string
# import preprocessor as p
# from preprocessor.api import clean, tokenize, parse
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import pandas as pd
import warnings

warnings.filterwarnings('ignore', '.*')

nlp = spacy.load("en_core_web_lg")
# nlp = spacy.load("en_vectors_web_lg")


translator = str.maketrans('', '', string.punctuation) 
stop_words = set(stopwords.words("english")) 

def dataPreprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text) 
    text =  text.translate(translator) 
    text = " ".join(text.split()) 

    text = re.sub("[^a-zA-Z]+", " ", text)

    word_tokens = word_tokenize(text) 
    text = [word for word in word_tokens if word not in stop_words] 
    text = [lemmatizer.lemmatize(word) for word in text] 

    # text = [word for word in text if word not in nlp.vocab[word].vector] 
    # print()
    return text


categories = [["health", "Medicine"], ["Business", "Finance"], ["Education", "school"], ["Government", "Politics"], ["Travelling", "Tourism"], ["Religion", "Beliefs"],["Agriculture", "Feeding"], ["Science", "Technology"],["Race", "Ethnicity"]]
# print(nlp('Religion').similarity(nlp('Beliefs')))

# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()

new_categotries = []

for category in categories:
    new_categotries.append([lemmatizer.lemmatize(category[0].lower()), lemmatizer.lemmatize(category[1].lower())])

print(categories)
print("===============================")
print(new_categotries)
# Process whole documents
text = ("When Sebastian Thrun started working on self-driving cars at "
        "Google in 2007, Profit few people outside of the company took him "
        "seriously. “I can tell you very senior CEOs of major American "
        "car companies would shake my hand and turn away because I wasn’t "
        "worth talking to,” said Thrun, in an interview with Recode earlier "
        "this week.")

df = pd.read_csv('data/newDfList.csv')

print(df.head(5))


print(df.columns)
sys.exit(0)
for i in range(len(df)):
    print(df.loc[i, "sn"])

clean_data  = dataPreprocess(text)

sys.exit(0)
avg_array = []
for new_cat in new_categotries:
    # print(text + " :  "  + new_cat[0] + " , " + new_cat[1] + " = " + str(nlp(text).similarity(nlp(new_cat[0]))) + " , " + str(nlp(text).similarity(nlp(new_cat[1]))))

    avg_score = []
    for index, text in enumerate(clean_data):
            
        cat1 = new_cat[0]
        cat2 = new_cat[0]
        
        similarity1 = nlp(text).similarity(nlp(new_cat[0]))
        similarity2 = nlp(text).similarity(nlp(new_cat[1]))

        if similarity1 > 0.5:
            print(similarity1)
            break;

        if similarity2 > 0.5:
            print(similarity2)
            break;

        # print(typ(similarity1))

        # individual_avgscore = (similarity1 + similarity2) / 2  

        # if individual_avgscore > 0.5:
        #     print(individual_avgscore)
        #     break;
        # if(index == (len(clean_data) -1 )):
        #     print(0)