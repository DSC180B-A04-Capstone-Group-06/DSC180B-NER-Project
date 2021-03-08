# DSC180B-NER-Project
This project focuses on the task of document classification using a BBC News Dataset. We will implement various classification models and compare the results to learn about the advantages and shortcomings of each method.

## Webpage
* https://dsc180b-a04-capstone-group-06.github.io/News-Classification-Webpage/

## Datasets Used
*BBC news: https://www.kaggle.com/pariza/bbc-news-summary</br>
*20 news group: http://qwone.com/~jason/20Newsgroups/
## Environment Required
* Please use the docker image: ``` littlestone111/dsc180b-ner-project  ```

## Run
```
$ launch-180.sh -i littlestone111/dsc180b-ner-project -G [group]
$ python run.py [test] [eda] 
```
```$ python run.py test``` will build the Bag-Of-Word and Tf-Idf models on the small test dataset and save the models to the model folder.
* ```BoG_model.pkl```: the parameter of Bag-Of-Word model.
* ```Tfidf_model.pkl```: the parameter of Tf-Idf model.

## Group Members
* Rachel Ung
* Siyu Dong
* Yang Li

## Our Findings
[Report TBA]




