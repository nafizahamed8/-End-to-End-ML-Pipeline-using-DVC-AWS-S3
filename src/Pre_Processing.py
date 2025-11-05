import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

#ensuring
log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

#setting up logger
logger=logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path=os.path.join(log_dir,"data_preprocessing.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """Transform the input text by converting it to lowercase ,tokenizing ,removing stopwords and punctuation ,and stemming"""

    ps=PorterStemmer()
    #convert to lowercase
    text=text.lower()
    text=nltk.word_tokenize(text)
    #removing non-alpha numeric value
    text=[word for word in text if word.isalnum()]
    #removing stopword and puncuation

    text=[ word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    #stem the words
    text=[ps.stem(word) for word in text]
    #join the token back into single string

    return " ".join(text)

def preprocess_df(df,text_column='text',target_column='target'):
    """Preprocess
    the dataframe by encoding the target column ,removing duplicates and transforming the text column"""

    try:
        logger.debug("starting preprocessing for dataframe")
        #Encode the target column
        encoder=LabelEncoder()
        df [target_column]=encoder.fit_transform(df[target_column])
        logger.debug("Target column Encoded")

        #remove duplicate rows
        df=df.drop_duplicates(keep="first")
        logger.debug("Duplicates removed")

        #Apply text transformation to the specified text column
        df.loc[:,text_column]=df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        return df
    
    except KeyError as e:
        logger.error("column not found %s:",e)

    except Exception as e:
        logger.error("Error during text normalization",e)
        raise

def main(text_column='text',target_column='target'):
    """
    Main function to load raw data,preprocess it and save the process data"""
    try:
        #Fetch the data from data/raw
        train_data=pd.read_csv('data/raw/train_csv')
        test_data=pd.read_csv('data/raw/test_csv')
        logger.debug('data loaded properly')

        #transform the data
        train_process_data=preprocess_df(train_data,text_column,target_column)
        test_process_data=preprocess_df(test_data,text_column,target_column)

        #store data inside data/process
        data_path=os.path.join("./data","interim")
        os.makedirs(data_path,exist_ok=True)

        train_process_data.to_csv(os.path.join(data_path,'train_process_csv'),index=False)
        test_process_data.to_csv(os.path.join(data_path,'test_process_csv'),index=False)

        logger.debug('process data save to %s',data_path)


    except FileNotFoundError as e:
        logger.error("file not found",e)
    except pd.errors.EmptyDataError as e:
        logger.error("no data: %s",e)
    except Exception as e:
        logger.error("failed to complete the data transformation process: %s",e)
        print(f"error:{e}")

if __name__=="__main__":
    main()

