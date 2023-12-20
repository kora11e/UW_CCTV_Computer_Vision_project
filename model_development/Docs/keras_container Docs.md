# keras_container Docs

## Class Variables

| *Name*      | *Type*              | *Description*                                                                                         |
| ----------- | ------------------- | ----------------------------------------------------------------------------------------------------- |
| data_path   | pathlib.Path        | Path to image data                                                                                    |
| img_size    | tuple               | Tuple of image dimensions in pixels                                                                    |
| split       | tuple               | Tuple with training data on index 0 and test data on index 1                                          |
| class_names | list                | List of class names generated by retrieving names of child directories of `./model_development/data/` |
| model       | tf.keras.Sequential | keras.Sequential model with layers and params set in `self.train_model()`                             |
| history     | tf.keras.callbacks.History | training logs generated in `self.train_model()`                                                       |
| predictions | tf.estimator.Estimator.predict   | Predictions with test data. set in `self.predict()`                                                   |

## Class Methods

### data_split()

    Splits the raw data set to training and test partitions with the use of `keras.utils.image_dataset_from_directory()`.

    Params:

+ `data_dir: Path`-> Path to image data directory
+ `img_size: tuple`-> tuple of image dimensions (defined in initialization)
+ `test_split: float = 0.2` -> percentage of rows used for testing (80/20 default)
+ `batch_size: int = 64`-> size of batch partitions for training and test data sets
+ `seed: int = np.random.randint(10000000, 99999999)`-> random seed shared by training and test data sets

    Return -> `tuple(training_set, test_set)`

### train_model()

    Initiates `tf.keras.Sequential` model and trains if prompted.

    Params:

+ `epoch: int = 1000` -> epoch quantity used for training
  
  `metrics: list = []`-> list of metrics to be printed during training. Will not pass into model.compile if it is empty
  
  Return -> `None`

### predict()

    makes prediction with test data if training is completed

    Return -> `None`



### get_history()

    Returns history for access to training logs.

    Return -> `self.history`

### get_predictions

    Returns predictions.

    Return -> `self.predictions`

## Notes

    Class methods will be tested and bug fixed when database is implemented and we have access to the data directories that we'll actually utilize in the project. As of now the class methods and structure are a template and are subject to change of any scale.