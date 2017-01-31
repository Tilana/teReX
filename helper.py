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

def generateVocabulary(data):
	sentences = flattenList(data)
	vocabularySet = getWordSetOfList(sentences)
        vocabList = list(vocabularySet)
        vocabList.sort()
	return createDictionary(vocabList)

def getWordSetOfList(wordList):
	wordSet = set()
	for words in wordList:
		wordSet.update(words)
	return wordSet
		

def flattenList(multiLevelList):
	return [sum(elem, []) for elem in multiLevelList]

def createDictionary(List):
	mapping = zip(List, range(len(List)))
	return dict(mapping)
	
