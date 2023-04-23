import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf 
from PIL import Image
import os 
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

#Lista z danymi (obrazami w postaci array)
data = []
#Lista z typami znaków
labels = []
b = 1
#Liczba podfolderów (od 0 do 42) dostępnych w folderze 'train', każdy podfolder reprezentuje inną klasę
classes = 43 
cur_path = os.getcwd() #pobieramy aktualną ścieżkę
#Bierzemy obrazki i ich labele
for i in range(classes): 
  path = os.path.join(cur_path,'train', str(i)) 
  images = os.listdir(path) #lista zawartości każdego z podfolderów folderu train (czyli obrazki)
  for a in images:
    try:
      image = Image.open(path + "\\" + a)
      image = image.resize((30,30)) 
      image = np.array(image) 
      data.append(image) 
      labels.append(i) #każdy znak ma przypisany odpowiadający mu label (typ znaku np. ograniczenie 20km/h)
    except:
      print("Error loading image")                   
#zamieniamy na numpy array
data = np.array(data)
labels = np.array(labels)
#printujemy: dla data - liczbę obrazków[39209], wymiary [30x30] oraz info że są kolorowe [3] 
# dla labels - liczbę labeli w przypisaniu do obrazków (czyli różnych labeli jest tylko 43, ale obrazek ma przypisany do siebie label)
print(data.shape, labels.shape)          
#Rozdzielamy datasety obrazków w taki sposób, że labels będą targetem (czyli to co chcemy przewidzieć [poprawność określania labeli])
# test_size oznacza że do testów przydzielamy 20% datasetu, reszta idzie do treningu
# random_state - coś z shufflingiem danych, dla wielu wykonywań lepiej chyba zostawić tak jak jest
# x to rozdział data, y to rozdział labels
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print(X_train.shape,X_test.shape, y_train.shape, y_test.shape)
#Konwersja labels na macierze binarna - każdej odmiennej wartości odpowiada inny wiersz np. 1,0,0,0. Jeżeli wartość int się powtarza, to powtarza się również wiersz 1,0,0,0 w macierzy. Kolumn jest tyle ile różnych wartości (czyli w naszym przypadku 43) a wierszy tyle ile wartości łącznie (czyli ponad 31 tys. dla y_t1 i ponad 7 tys. dla y_t2)
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

#Budowa modelu CNN w celu podziału obrazów do odpowiednich kategorii

model = Sequential() #tworzymy nowy model sekwencyjny

#Dodajemy warstwę Conv2D - często używaną do analizy obrazów
#filters - liczba filtrów wyjściowych w warstwie
#kernel_size - rozmiar filtra splotowego
#activation - ustawia funkcję aktywacji do warstwy, w tym przypadku jest to relu
#input_shape - ustawia kształt danych wejściowych dla warstwy, wywnioskowany z kształtu danych treningowych 
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))

#Dodajemy warstwę maxpooling
model.add(MaxPool2D(pool_size=(2, 2)))
#Dodajemy warstwę Dropout - technika regularyzacji stosowana w celu zapobiegania nadmiernemu dopasowaniu. Rate ustawia ułamek danych wejściowych do losowego spoadku w czasie trenowania
model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))

#Dodajemy warstwę Flatten, która spłaszcza wyjście poprzedniej warsty na wektor 1D, którego można użyć jako wejście do w pełni połączonej warstwy
model.add(Flatten())

#Dodajemy warstwę Dense. Jest to warstwa która może uczyć się złożonych relacji pomiędzy featurami (czyli X)
#256 to liczba neuronów w warstwie
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
#Dodajemy warstwę Dense z funkcją softmax, która jest zwyczajowo używana do klasyfikacji problemów złożonych z wielu klas.
#Output tej warstwy reprezentuje prawdopodobieństwo każdej klasy
model.add(Dense(43, activation='softmax'))

#Kompilacja modelu
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Trenowanie modelu i walidacja
#Po zbudowaniu modelu architektury, używamy metody model.fit()
#Po 15 epokach i korzystając z 64 próbek?(spośród ponad 31 tys., ale nie wiem czy dobrze to rozumiem)
#otrzymujemy accuracy na poziomie 95%

#Na ten moment z jakiegoś powodu chyba bierze 491 sampli zamiast tych 31 tys., loss: 0,3361 accuracy:0.8970 val_loss:0.1174 val_accuracy:0.9689
eps = 15
anc = model.fit(X_train, y_train, batch_size = 64, epochs=eps, validation_data=(X_test, y_test))
print("siema")