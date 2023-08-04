import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

class Classifier:
    def __init__(self):
        self.board = cv.imread('Data/board.jpeg') # tablero original
        self.board_bin = self.binarizacion(self.board) # binarizacion
        self.boxes = self.slice(self.board_bin, 8, 8) # imagen de cada casilla del tablero

        #----------- FICHAS -----------
        self.datos = []
        self.tags = []
        
        # piezas negras
        tn = self.boxes[0]
        self.datos.append(self.areas_3x3(tn)) # se obtiene el vector de areas
        self.tags.append('B Rook') # se asigna su respectiva etiqueta
        cn = self.boxes[1]
        self.datos.append(self.areas_3x3(cn))
        self.tags.append('B Knight')
        an = self.boxes[2]
        self.datos.append(self.areas_3x3(an))
        self.tags.append('B Bishop')
        qn = self.boxes[3]
        self.datos.append(self.areas_3x3(qn))
        self.tags.append('B Queen')
        kn = self.boxes[4]
        self.datos.append(self.areas_3x3(kn))
        self.tags.append('B King')
        pn = self.boxes[8]
        self.datos.append(self.areas_3x3(pn))
        self.tags.append('B Pawn')

        # piezas blancas
        tb = self.boxes[56]
        self.datos.append(self.areas_3x3(tb))
        self.tags.append('W Rook')
        cb = self.boxes[57]
        self.datos.append(self.areas_3x3(cb))
        self.tags.append('W Knight')
        ab = self.boxes[58]
        self.datos.append(self.areas_3x3(ab))
        self.tags.append('W Bishop')
        qb = self.boxes[59]
        self.datos.append(self.areas_3x3(qb))
        self.tags.append('W Queen')
        kb = self.boxes[60]
        self.datos.append(self.areas_3x3(kb))
        self.tags.append('W King')
        pb = self.boxes[48]
        self.datos.append(self.areas_3x3(pb))
        self.tags.append('W Pawn')

        # vacio
        self.datos.append(np.array([0,0,0,0,0,0,0,0,0]))
        self.tags.append('-')
        
        self.datos = np.array(self.datos)

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
    
    def euclideanD(self, x1, x2): # metrica euclideana
        return np.sqrt(np.sum((x1-x2)**2))    

    def predict(self, path):
        board = cv.imread(path) # lectura de la imagen
        board_bin = self.binarizacion(board) # binarizacion
        boxes = self.slice(board_bin, 8, 8) # division de las casillas del tablero

        # calculamos los vectores de areas de cada box
        areas = []
        for box in boxes:
            areas.append(self.areas_3x3(box))
        areas = np.array(areas)

        # buscamos el vecinos mas cercanos de cada box (kNN)
        plot_tags = []
        for i in areas:
            distances = np.array([self.euclideanD(i, di) for di in self.datos])
            ni = np.argsort(distances)[0] # indice del mas cercanos
            plot_tags.append(self.tags[ni]) # etiqueta de la prediccion

        self.plot_grid(boxes, 8, 8, plot_tags) # graficamos las predicciones de cada pieza

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

    print(clasificador.datos)
    clasificador.predict('imgs/test2.jpeg')