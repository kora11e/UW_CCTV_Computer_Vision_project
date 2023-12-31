import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import numpy as np

class KERAS_container():
    def __init__(self, data_path:Path, img_size:tuple) -> None:

        self.data_path = data_path # Path to image data
        self.img_size = img_size # Image size in pixels
                
        self.split = self.data_split(data_dir=self.data_path, img_size=img_size) # Train/Test split in tuple
        
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
                                                        seed=seed,
                                                        image_size=img_size,
                                                        batch_size=batch_size
                                                        )
        self.class_names = train.class_names # list of classification categories (retrieved from database directory names)
        
        # DataSet performance optimizations and train set shuffling
        train = train.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        test = test.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
        
        # Seed for result repeatability, logging seeds allows keeping shuffle as the only random variable if need be.
        print(f"Seed: {seed}")
        
        return (train, test)
    
    # Training initializer
    def train_model(self, epoch=1000, metrics:list=[]):
        
        ds = self.split
        self.model = keras.models.Sequential([ # Layers are in default config, modifications may be applied as project moves forward
                                            keras.layers.Rescaling(1./255, input_shape=(self.img_size[0], self.img_size[1], 3)),
                                            keras.layers.Conv2D(16, 3, padding='same', activation='softmax'),
                                            keras.layers.MaxPooling2D(),
                                            keras.layers.Conv2D(32, 3, padding='same', activation='softmax'),
                                            keras.layers.MaxPooling2D(),
                                            keras.layers.Conv2D(64, 3, padding='same', activation='softmax'),
                                            keras.layers.MaxPooling2D(),
                                            keras.layers.Flatten(),
                                            keras.layers.Dense(128, activation='softmax'),
                                            keras.layers.Dense(len(self.class_names))
                                            ])
        compile_params = {
                        'optimizer': 'SDG',
                        'loss': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                        }
        if metrics:
            compile_params['metrics'] = metrics
            
        self.model.compile(**compile_params)
        
        
        print(self.model.summary())
       
        while True: # User confirmation to avoid accidential compute time usage
            conf = input('Confirm training initiation (y/n): ')
            if conf in ('y', 'n'):
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

        match conf:
            case 'y':
                self.history = self.model.fit(
                    ds[0],
                    validation_data=ds[1],
                    epochs=epoch
                )
            case 'n':
                print('Training aborted.')
        
    # Predictor
    def predict(self):
        if self.history:
            self.predictions = self.model.predict(self.split[1])
        else:
            raise ValueError("Untrained Model: Initiate model training through model_train() before running predictions.")
    
    # Getter methods    
    def get_history(self):
        return self.history
    
    def get_pred(self):
        return self.predictions
        