# keras_container Docs

## Class Variables

| *Name* | *Type* | *Description* |
| --- | --- | --- |
| data | pd.DataFrame | Raw data container |
| train | pd.DataFrame | Training split of  `self.data` |
| test | pd.DataFrame | Test split of `self.data` |
| train_data | pd.DataFrame | Feature split of `self.train` |
| y_train | pd.DataFrame | Label split of `self.train` |
| test_data | pd.DataFrame | Feature split of `self.test` |
| y_test | pd.DataFrame | Label split of `self.test` |
| model | tf.keras.Sequential or None | keras.Sequential model with layers and params set in `self.train_model()` |
| predictions | list(array[pred_for_index_0],                  array[pred_for_index_1], ...) | Predictions with test data. set in `self.predict()` |

## Class Methods

### data_split()

    Utilizes `sklearn.model_selection.train_test_split()` to set `self.train` and `self.test`.

    Params:

+ `test_split: float = 0.2` -> percentage of rows used for testing (80/20 default)
  

    Return dtype -> `None`

### feature_label_split()

    Splits train or test data sets into features and labels, setter of `self.train_data` and `self.y_train`.

    Params:

+ `labels: list` -> list of label columns as strings
  
+ `train= True` -> when true, sets for training data, else returns test data.
  

    Return dtype -> `None | (test_data, y_test)`

### train_model()

    Initiates model training, outputs final loss in print statement.

    Params:

+ `l1=30` -> size of 1st layer
  
+ `l2=10` -> size of 2nd layer
  
+ `epoch= 1000` -> epoch quantity used for training
  
+ `batch_size= 128` -> no of elements per batch of input
  
+ `v= 2` -> verbose parameter of model.fit() (2= 1-line print statement) 
  
  Return dtype -> `None`
## Notes

    This class is configured for numerical data, therefore reshaping of images will be required.

    Model depth and layer densities can be optimized through optimizer functions on implementation. Default values are assigned to model variables until said functions are implemented and we populate the database.
