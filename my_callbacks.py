import keras

class Histories(keras.callbacks.Callback):
    def __init__(self,display=100):
        '''
        display: Number of batches to wait before outputting loss
        '''
        self.seen = 0
        self.display = display
        self.epch = 0

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.seen = 0
        self.epch += 1
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        self.seen += logs.get('size', 0)
        if self.seen % self.display == 0:   
            scores = self.model.evaluate([self.model.validation_data[0], self.model.validation_data[1]], self.model.validation_data[2], verbose=0)
            print('Epoch {0} - Batch {1}/{2} - Batch Acc: {3}'.format(self.epch, self.seen, self.params['nb_sample'], scores[1]))
        return
