import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import os


def create_output(iname= 17):
	if not(os.path.exists('./SampleOutput')):
		os.mkdir('./SampleOutput')
	compact_model = load_model('./ModelWeights/ccnn_weights.h5')
	I = cv2.imread('../Class8/Test/Image/{:04d}.PNG'.format(iname),cv2.IMREAD_GRAYSCALE).reshape((1,512,512,1))
	mask = cv2.imread('../Class8/Test/Label/{:04d}_label.PNG'.format(iname))
	seg, score = compact_model.predict(I)
	plt.figure(figsize=(18,10))
	plt.subplot(131)
	plt.imshow(I[0].reshape(512,512))
	plt.subplot(132)
	segmap = seg[0][:][:].reshape((128,128))
	plt.imshow(segmap.astype('uint8'))
	props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
	plt.text(0.2, 0.1, f'Score={score[0]}', fontsize=14,
	        verticalalignment='top', bbox=props)
	plt.subplot(133)
	plt.imshow(mask)
	plt.show()
	# plt.savefig(f'./SampleOutput/output{iname}.png',bbox_inches='tight')
