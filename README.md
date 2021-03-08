# DSC180B-NER-Project
This project focuses on the task of document classification using a BBC News Dataset and a 20 News Group Dataset. We will implement various classification models and compare the results to learn about the advantages and shortcomings of each method.

## Webpage
* https://dsc180b-a04-capstone-group-06.github.io/News-Classification-Webpage/

## Datasets Used
* BBC news: https://www.kaggle.com/pariza/bbc-news-summary Download this dataset</br>
* 20 news group: http://qwone.com/~jason/20Newsgroups/ This dataset is fetch by using sklearn package
## Environment Required
* Please use the docker image: ``` littlestone111/dsc180b-ner-project  ```

## Run
```
$ launch-180.sh -i littlestone111/dsc180b-ner-project -G [group]
$ python run.py [all] [preprocessing] [autophrase] [model]
```
```all``` target will run everthing inlcuded in project, and return the final prediction on the test dataset for document classification.</br>
```preprocessing``` target will preprocess 20 news group data for AutoPhrase, so that they can be used for training the model.</br>
```autophrase```target will run Professor Shang's Autophrase model to extract quality phrases from the dataset.</br>
```model```target will build the SVM+ALL+TF-IDF combined vocab list model for 20 news group dataset. </br>
```$ python run.py test``` will build the Tf-Idf models on the small test dataset and save the models to the model folder.
* ```model.pkl```: the parameter of the final model.

## Group Members
* Rachel Ung
* Siyu Dong
* Yang Li

## Our Findings
The BERT classification on the 5 news BBC datasetdoes not over perform any of our implemented mod-els.  From our result table, we observed that all of ourmodels have performance on F1 and accuracy around.95,  which  means  all  of  them  are  useful  and  power-ful.    The  best  of  them  is  the  SVM+ALL(TF-IDF),which uses the vocab from both NER result and Au-toPhrase result. This is expected, because the name en-tities and high quality phrases are actually different indifferent fields, which makes them become good fea-tures for prediction.   For the 20 News group dataset,the SVM+ALL(TF-IDF) also over performed the othermodels at F1 and Accuracy being .84. Considering theclasses are huge(20), this result is useful and powerful.Applying our best model on the 5 news BBC datasetwe got F1 at 0.9525, and accuracy at 9528, for the 20news group we got F1 at .8463 and accuracy at .8478.




