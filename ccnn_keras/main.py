from training import train
from model import compact_cnn_model
from sample_output import create_output

segtrainpath = './Class8/Train'
segvalpath =  './Class8/Test'
classtrainpath = './Class8/Classification/Train'
classvalpath = './Class8/Classification/Test'

train(segtrainpath,segvalpath,classtrainpath,classvalpath)
print('Training done')


#Create the compact network
model = compact_cnn_model('./ModelWeights/seg_weights.h5','./ModelWeights/class_weights.h5')
print(model.summary())

#Save the model
model.save('./ModelWeights/ccnn_weights.h5')

#Visualise output
create_output()