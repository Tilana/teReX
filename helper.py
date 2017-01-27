def createFilename(path, RBF, conversion, cosSim, Laplace):
	
	name='graph'
	if RBF:
		name=name + '_RBF'
	if conversion is not None:
		name = name + '_' + conversion
	if cosSim:
		name = name + '_cos'
	if Laplace:
		name = name + 'Laplace'
        filename = 'results/'+path.split('/')[1].split('.')[0]+'/'+name+'.txt'
	return filename
