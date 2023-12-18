import tensorflow as tf
from sklearn.model_selection import train_test_split

class KERAS_container():
    def __init__(self, data) -> None:

        self.data = data # Raw DataFrame object
        
        self.train = None # Training split
        self.test = None # Test Split
        
        # Feature/Label split
        self.train_data = None
        self.y_train = None
        
        
        # Model variables
        self.model = None # tf.keras.Sequential(params)
        self.predictons = None # model.predict() -> [np.array[precdiction0], np.array[prediction1], ...]
        
    # Train/Test splitter
    def data_split(self, test_split:float= 0.2):
        self.train, self.test = train_test_split(self.data, test_size=test_split, shuffle=False) # 80/20 split. shuffle = False, tensorflow's shuffle is implemented in input_fn
    
    # Feature/Label splitter
    def feature_label_split(self, labels:list, train= True):
        if train:
            self.y_train = self.train.loc[:, labels]
            self.train_data = self.train.drop(columns=labels, inplace=False)
        else:
            y_test = self.test.loc[:, labels]
            test_data = self.test.drop(columns=labels, inplace=False)
            
            return (test_data, y_test)

    # Training initiator
    def train_model(self, l1=30, l2=10, epoch= 1000, batch_size= 128, v= 2):
        self.model = tf.keras.Sequential(layers=[
            tf.keras.layers.Dense(l1, activation='softmax', input_shape=(len(self.feature_cols),)),
            tf.keras.layers.Dense(l2, activation='softmax'),
            tf.keras.layers.Dense(1)  # Output layer
        ])
        
        self.model.compile(optimizer='SDG', loss='mean_squared_error')
        self.model.fit(self.train, self.y_train, epochs=epoch, batch_size=batch_size, verbose=v)