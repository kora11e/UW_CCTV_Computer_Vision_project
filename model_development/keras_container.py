import tensorflow as tf
from tensorflow import keras

from keras import layers
from keras.models import Sequential

import numpy as np

from pathlib import Path
from utils.logger import Logger

class KERAS_container(Logger):
    def __init__(self, data_path:Path, img_shape:tuple) -> None:
        self.id = id(self)
        
        super().__init__(class_name='KERAS_container', instance_id = self.id)
        
        self.data_path = data_path # Path to image data
        self.img_shape = img_shape # Image size in pixels
                
        self.split = self.data_split(data_dir=self.data_path, img_size=(img_shape[0], img_shape[1])) # Train/Test split in tuple
        
    
    # Train/Test splitter
    def data_split(self, data_dir:Path, img_size:tuple, test_split:float= 0.2, batch_size:int= 64, seed:int= np.random.randint(10000000, 99999999)):
        train = keras.utils.image_dataset_from_directory(
                                                        directory=data_dir, 
                                                        validation_split=test_split, 
                                                        subset='training', 
                                                        seed=seed, 
                                                        image_size=img_size,
                                                        batch_size=batch_size
                                                        )
        
        test = keras.utils.image_dataset_from_directory(
                                                        directory=data_dir,
                                                        validation_split=test_split,
                                                        subset='validation',
                                                        seed=seed - np.random.randint(1000000, 9999999),
                                                        image_size=img_size,
                                                        batch_size=batch_size
                                                        )
        self.class_names = train.class_names # list of classification categories (retrieved from database directory names)
        
        # DataSet performance optimizations and train set shuffling
        train = train.cache().shuffle(1000, seed=seed).prefetch(buffer_size=tf.data.AUTOTUNE)
        test = test.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        
        '''Seed logging:
        Seed printed for result repeatability. Logging and manually setting a seed will omit random variables if need be.
        For perfect reproducability, tf.data.Dataset.shuffle() also utilizes the same seed as keras.utils.image_dataset_from_directory methods. 
        Omitting shuffle while training results are suboptimal is ill-advised. Seeds should be utilized to reproduce a pre-optimized training
        config without having to move a trained model in memory.
        '''
        self.log(f"Data Loaded with seed: {seed}")
        print(f"Seed: {seed}")
        return (train, test)
    
    # Training initializer
    def train_model(self, epoch=1000, metrics:list=[]):
        
        ds = self.split
        self.model = Sequential([ # Layers are in default config, modifications may be applied as project moves forward
                                layers.Rescaling(1./255, input_shape=self.img_shape),
                                layers.Conv2D(16, 3, padding='same', activation='relu'),
                                layers.MaxPooling2D(),
                                layers.Conv2D(32, 3, padding='same', activation='relu'),
                                layers.MaxPooling2D(),
                                layers.Conv2D(64, 3, padding='same', activation='relu'),
                                layers.MaxPooling2D(),
                                layers.Flatten(),                                    
                                layers.Dense(128, activation='softmax'),
                                layers.Dense(len(self.class_names)) # Output layer
                                ])
        compile_params = {
                        'optimizer': 'Adam',
                        'loss': keras.losses.SparseCategoricalCrossentropy()
                        }
        if metrics:
            compile_params['metrics'] = metrics
            
        self.model.compile(**compile_params)
        
        
        print(self.model.summary())
       
        while True: # User confirmation to avoid accidential compute time usage
            conf = input('Confirm training initiation (y/n): ').lower()
            if conf in ('y', 'n', 'yes', 'no'):
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

        match conf:
            case 'y', 'yes':
                self.log(f"Training initiated")
                self.history = self.model.fit(
                    ds[0],
                    validation_data=ds[1],
                    epochs=epoch
                )
            case 'n', 'no':
                print('Training aborted.')
        
        self.log(f"Training completed. Training summary: \n{self.get_history()}\n")
    # Predictor
    def predict(self):
        if self.history:
            self.log("Prediction initiated")
            self.predictions = self.model.predict(self.split[1])
            self.log("Prediction completed.")
        else:
            raise ValueError("Untrained Model: Initiate model training through model_train() before running predictions.")
    
    # Getter methods    
    get_history = lambda self: self.history.history
    get_preds = lambda self: self.predictions
        