# Chess pieces classification with area correlation
[Versión en español](README_ES.md)

In this repository, is presented a simple method to classify chess pieces on a board. Unlike traditional deep learning-based approaches, the problem is approached using area correlation and k-NN classifier, which allows us to implement a straightforward and intuitive method that performs well in controlled environments.

## Method description
The classification is carried out using the information of each piece's area. To gather such information, we work with boards from the page. [Chess](https://www.chess.com/es):

![alt](imgs/board.jpeg)

To build the dataset to work with _k-NN_ classifier, the board image is binarized, and the information of each square is divided:

![alt](imgs/Tablero.png)

Subsequently, for each piece, the image is divided into windows, and the number of black pixels in each window is counted:

![alt](imgs/KingArea.png)

Consequently, we obtain nine-entry vectors that represent the information of each piece.

Each board is processed using the method described above to calculate the area of each square, finally, *k-NN* is used to predict the most similar piece for each square and assign a label:

![alt](imgs/Prediccion.png)

## Instalation
Clone or download this repository:
```bash
git clone git@github.com:daniel-lima-lopez/Chess-pieces-classification-with-area-correlation.git
```

## Example
To instantiate the classifier, it is necessary to import the class:
```python
from ChessClassifier import Classifier

classifier = Classifier()
```

Subsequently, to make a prediction, you must include the location of the board to be classified.:
```python
classifier.predict('path')
```
Example [boards](test/) are included to test the classifier.