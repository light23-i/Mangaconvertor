import cv2
import os
import pytesseract
from boxdetector import detectFromPaths
import fittext
from googletrans import Translator
from fnmatch import fnmatch
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
translator = Translator()


def takeFilesByExtension(folder, pattern):
	paths = []
	for path, subdirs, files in os.walk(folder):
		for name in files:
			if fnmatch(name, pattern):
				paths.append(os.path.join(path, name))

	return paths

def translate(folder = 'test'):
	
	print('translator loaded')
	
	paths = takeFilesByExtension(folder, '*.jpg')
	print('paths taken')
	
	ballons, locations, pages = detectFromPaths(paths)
	
	path = pages[0]
	img = cv2.imread(path)
	
	for ballon, location, page in zip(ballons, locations, pages):
		if path != page:
			cv2.imwrite(os.path.join('translated',os.path.basename(path)), img)
			path = page
			img = cv2.imread(path)
		result = pytesseract.image_to_string(ballon)
		result = translator.translate(result)
		fited_text = fittext.fit_text(ballon.shape[0],ballon.shape[1],result)
		img[location[0]:location[1],location[2]:location[3]] = fited_text

if __name__=='__main__':
	translate()