from model import CompactCNN
import cv2
import matplotlib.pyplot as plt
import torch

#Check if you have a GPU and CUDA.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Create the model
model = CompactCNN().to(device)
#Load the saved model
modeldata = torch.load('./models/compactcnnweights')
#Initialize with the saved weights
model.load_state_dict(modeldata['model_state_dict'])
#Set model to evaluate mode
model.eval()

#Print which class label corresponds to what
#In this case 0 means defective image and 1 means defect free image
print('Prediction labels:' ,modeldata['classidx'])
#Open an image and predict 
ino=62
img = cv2.imread(f'../Class8/Test/Image/{ino:04d}.PNG',0).reshape(1,1,512,512)
a = model(torch.from_numpy(img).type(torch.cuda.FloatTensor)/255)
print(f'Prediction probability: {a[1].item():.3e}')
plt.figure(figsize=(10,10))
plt.subplot(121)
plt.imshow(img[0][0])
plt.subplot(122)
plt.imshow(a[0].cpu().detach().numpy()[0][0])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.2, 0.1, f'Score={a[1].item():03f}', fontsize=14,
        verticalalignment='top', bbox=props)
plt.axis('off')
plt.show()