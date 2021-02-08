# DSC180B-NER-Project

## Environment Required
* Please use the docker image: ``` littlestone111/dsc180b-ner-project  ```

## Run
```
$ launch-180.sh -i littlestone111/dsc180b-ner-project -G [group]
$ python run.py [-test]  [-eda] (For example: python run.py -test).
```
```python run.py -test``` will build the Bag-Of-Word and Tf-Idf models on the small test dataset and save the models to the model folder.
* ```BoG_model.pkl```: the parameter of Bag-Of-Word model.
* ```Tfidf_model.pkl```: the parameter of Tf-Idf model.


