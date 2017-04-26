from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import Callback

import numpy as np
import random
import sys

import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
# os.environ["THEANO FLAGS"] = 'mode=FAST_RUN, device=gpu, lib.cnmem=0.95, floatX=float32'

text_org = open('HumanAction.txt').read().lower()
print 'corpus length:', len(text_org)

#Unique characters
chars_org = sorted(list(set(text_org)))
print 'total chars:', len(chars_org)

#Unwanted characters
unwanted=['\n','\r','\x80', '\x84', '\x88', '\x89', '\x92', '\x93', '\x94', '\x96', '\x98', '\x99', '\x9c', '\x9d', '\x9e', '\x9f', '\xa0', '\xa1', '\xa3', '\xa4', '\xa6', '\xa7', '\xa8', '\xa9', '\xaa', '\xab', '\xae', '\xaf', '\xb6', '\xb7', '\xb9', '\xbb', '\xbc', '\xbe', '\xc2', '\xc3', '\xc5', '\xc6', '\xce', '\xcf', '\xe2']

print text_org[1:1000]
print chars_org

#removing unwanted characters and again merging it into string
text=[x for x in text_org if not x in unwanted]
text=''.join(text)
print len(text)

#Unique characters after redundant characters are removed
chars = sorted(list(set(text)))
print 'total chars:', len(chars)

#creating two one-to-one dictionaries mapping each unique character/number to unique number/character.
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

print text[:1000]
print chars

##cut the text in semi-redundant sequences of maxlen characters
maxlen = 50
#step with which the sequence is moved in the input
step = 5
sentences = []
next_chars = []

#generating all sequences of maxlen from the text file with step/stride=5  
for i in range(0, len(text)/6 - maxlen, step):
	sentences.append(text[i: i + maxlen])
	next_chars.append(text[i + maxlen])
print 'nb sequences:', len(sentences)

print 'Vectorization...'
#converting the sequences and the character to be predicted into one-hot encoding
#from the previous step
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
	for t, char in enumerate(sentence):
		X[i, t, char_indices[char]] = 1
	y[i, char_indices[next_chars[i]]] = 1


# build the model: using LSTM
#with return sequences the first layer of lstm returns all its output sequences
#2 layer LSTM network
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars)),return_sequences= True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

#Loading saved weights from previous training
#model.load_weights('lstm_weights.h5')


#Defining Optimizer and its parameters
#Clipping the gradients and learning rate is decayed with increase in epochs
#decay is not present in older version of keras
# optimizer = RMSprop(lr=0.002,clipnorm=100,decay=0.05)
optimizer = RMSprop(lr=0.002,clipnorm=100)
#Training function mentioing the type of loss function and optimizer used.
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#summarizes the parameters and output of each layer
print model.summary()

def sample(preds, temperature=1.0):
	# helper function to sample an index from a probability array
	preds = np.asarray(preds).astype('float64')
	#transforming the probability scores with the temperature
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	#chooses a character with the following distributed probabilities
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)

loss=[]
val_loss=[]

#Plotting graph of loss and validation loss over epochs
def plot_graph():
	plt.ylim([0,3])
	plt.xlim([0,len(loss)])
	plt.plot(loss,label="Loss")
	plt.plot(val_loss,label="Validation loss")
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epochs')
	plt.savefig('plot_'+str(datetime.datetime.now()).split(' ')[1].split('.')[0]+'.jpg')
	plt.show()

#randomly chooses a start sequence and starts predicting the output character while moving 
#through the new generated sequence
def generate_text(iteration,curr_loss,curr_val_loss):
	#randomly selecting the first sequence to start with
	start_index = random.randint(0, len(text) - maxlen - 1)
	#generates text for different temperaures
	for diversity in [0.2,0.5,1.0]:
		print ''
		print '----- diversity:', diversity

		generated = ''
		sentence = text[start_index: start_index + maxlen]
		generated += sentence
		print '----- Generating with seed: "' + sentence + '"'
		sys.stdout.write(generated)

		#first predicts the next character for a given sequence, then discards the first character 
		#and includes the predicted character. This is performed in a loop generating a sequence of length of 400
		for i in range(400):
			#converts the randomly sampled sequence into One-hot encoding
			x = np.zeros((1, maxlen, len(chars)))
			for t, char in enumerate(sentence):
				x[0, t, char_indices[char]] = 1.

			#predicts probability scores of next character for the built model
			preds = model.predict(x, verbose=0)[0]
			#transforms the predicted probabilities using temperature and selects the next character 
			next_index = sample(preds, diversity)
			next_char = indices_char[next_index]
			generated += next_char
			#the first character is discarded and the predicted character is appended at the end
			sentence = sentence[1:] + next_char

			sys.stdout.write(next_char)
			sys.stdout.flush()
		print ''
		#The generated text for temperature=0.5 is saved it into a text file as it is more consistent
		if diversity==0.5:
			with open('generated.txt','a') as f:
				f.write('Epoch Number: '+str(iteration)+'  Loss: '+str(curr_loss)+' Val_Loss: '+str(curr_val_loss)+'\n')
				f.write('----- Generating with seed: "' + sentence + '"' + '\n')
				f.write(sentence+generated)
				f.write('\n')
				f.write('\n')

#Saves the weights at the end of each epoch
class do_epoch(Callback):
    def on_epoch_end(self, epoch, logs={}):
        model.save_weights('lstm_weights.h5',overwrite=True)
        #metrics after each epoch are saved into a list
        loss.append(logs['loss'])
        val_loss.append(logs['val_loss'])
        #the generate text function is called after each epoch
        generate_text(epoch,logs['loss'],logs['val_loss'])
		#separately saves the weights for every 6 epochs
        if epoch%6==0: 
        	model.save_weights('lstm_weights_'+str(epoch)+'.h5',overwrite=True)

#callback function
checkpointer = do_epoch()

try:
	# train the model, output generated text after each epoch
	model.fit(X, y, batch_size=64,validation_split=0.1, nb_epoch=100,callbacks=[checkpointer])
	#plot the metrics after completing training
	plot_graph()

except KeyboardInterrupt:
	#summarize history for loss if program is interrupted
	plot_graph()