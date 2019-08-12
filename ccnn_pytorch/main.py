from model import CompactCNN
import os
from trainer import train_class_model,train_seg_model
from datahandler import classification_dataset,segmentation_dataset
import torch
import torch.nn

def main():
	# Create the CompactCNN model
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = CompactCNN().to(device)

	# Train for the segmentation 
	criterion = torch.nn.MSELoss(reduction='mean')
	optimizer = torch.optim.Adadelta(list(model.parameters())[:-6])
	dataloaders=segmentation_dataset()
	trained_model = train_seg_model(model,criterion, optimizer,dataloaders,device,num_epochs=25)

	if not os.path.isdir('./models'):
		os.mkdir('./models')
	torch.save(trained_model.state_dict(),'./models/segmentationweights')

	dataloaders,trainmap,testmap = classification_dataset()

	# model.load_state_dict(torch.load('./models/segmentationweights'))
	criterion = torch.nn.BCELoss()
	optimizer = torch.optim.Adadelta(list(model.parameters())[-6:])
	trained_model = train_class_model(model, criterion, optimizer,dataloaders,device,num_epochs=10)
	torch.save({'classidx':trainmap,'model_state_dict':trained_model.state_dict()},'./models/compactcnnweights')

if __name__ == '__main__':
	main()