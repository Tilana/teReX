import os

def createFilename(path, RBF, conversion, cosSim, normalize):
	
	name='graph'
	if RBF:
		name=name + '_RBF'
	if conversion is not None:
		name = name + '_' + conversion
	if cosSim:
		name = name + '_cos'
	if normalize:
		name = name + '_renorm'
	newPath = 'results/'+path
	createDirectory(newPath)
        filename = newPath +'/'+name+'.txt'
	return filename


def createDirectory(path):
	if not os.path.exists(path):
		os.makedirs(path)

