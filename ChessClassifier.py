import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from sklearn.neighbors import NearestNeighbors

class Classifier:
    def __init__(self):
        self.board = cv.imread('Data/board.jpeg') # tablero original
        self.board_bin = self.binarizacion(self.board) # binarizacion
        self.boxes = self.slice(self.board_bin, 8, 8) # imagen de cada casilla del tablero

        #----------- FICHAS -----------
        self.datos = []
        self.tags = ['B Rook', 'B Knight', 'B Bishop', 'B Queen', 'B King', 'B Pawn',
                     'W Rook', 'W Knight', 'W Bishop', 'W Queen', 'W King', 'W Pawn', '-']
        for i in [0,1,2,3,4,8,56,57,58,59,60,48]:
            self.datos.append(self.areas_3x3(self.boxes[i]))
        self.datos.append(np.array([0,0,0,0,0,0,0,0,0]))
        self.datos = np.array(self.datos)
        self.y = np.arange(0,13)

        #----------- KNN ---------------
        self.kNN = NearestNeighbors(n_neighbors=1)
        self.kNN.fit(self.datos, self.y)        

    def binarizacion(self, matrix):
        gray = cv.cvtColor(matrix, cv.COLOR_BGR2GRAY) # escala de grises
        bin = np.zeros(shape=(matrix.shape[0], matrix.shape[1]), dtype=np.uint8) # se crea una matriz de ceros
        # se recorre toda la matriz de la imagen
        for ir, row in enumerate(gray):
            for ic, pixel in enumerate(row):
                if pixel > 100: # umbral de binarizacion
                    bin[ir, ic] = 255 # si se pasa el umbral se pinta de blanco
        return bin

    def slice(self, img, rows, cols):
        dr = int(img.shape[0]/rows) # longitud de cada slice
        dc = int(img.shape[1]/cols) # altura de cada slice
        imgs = [] # lista de slices
        for i in range(rows):
            for j in range(cols):
                imgs.append(img[i*dr : (i+1)*dr, j*dc : (j+1)*dc])
        return imgs
    
    def areas_3x3(self, box):
        grid = self.slice(box, 3, 3) # se divide la imagen de entrada en una lista de 9 elementos
        aux = [] # vector de areas de cada slice
        # se recorre la lista de slices
        for slice in grid:
            aux_area = 0
            for ir in range(slice.shape[0]):
                for ic in range(slice.shape[1]):
                    if slice[ir, ic]==0: # buscamos pixeles negros
                        aux_area += 1 # incrementa el conteo de area
            aux.append(aux_area)
        return np.array(aux)

    def predict(self, path):
        board = cv.imread(path) # lectura de la imagen
        board_bin = self.binarizacion(board) # binarizacion
        boxes = self.slice(board_bin, 8, 8) # division de las casillas del tablero

        # calculamos los vectores de areas de cada box
        areas = []
        for box in boxes:
            areas.append(self.areas_3x3(box))
        areas = np.array(areas)

        # predicciones con kNN
        preds = self.kNN.kneighbors(areas, return_distance=False)
        tags = [self.tags[int(i)] for i in preds]

        self.plot_grid(boxes, 8, 8, tags) # graficamos las predicciones de cada pieza

    def plot_grid(self, imgs, rows, cols, tags = []):
        fig, axes = plt.subplots(rows, cols)
        fig.set_size_inches(13.5, 13.5, forward=True)
        for i, img in enumerate(imgs):
            aux = plt.subplot(rows, cols, i+1)
            aux.get_xaxis().set_visible(False)
            aux.get_yaxis().set_visible(False)
            aux.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            if tags != []:
                aux.set_title(tags[i], fontsize = 11) # prediccion de cada casilla
        fig.savefig('prediccion', dpi=100, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    clasificador = Classifier()

    clasificador = Classifier()
    clasificador.predict('test/test6.jpeg')