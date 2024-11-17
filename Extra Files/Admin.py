#Modules

from tkinter import messagebox, simpledialog, filedialog, ttk
from tkinter import *
import tkinter
from math import *
from imblearn.over_sampling import RandomOverSampler 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os, cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D
from PIL import Image


main = Tk()
main.title("Skin Disease Diagnosis Using Convolutional Neural Network")
main.geometry("1300x1200")
#main.config(bg="powder blue")
main.config(bg="LightBlue1")

global filename
global X, Y
global x, y
global X_train, X_test, y_train, y_test
global accuracy
global dataset
global model

#Functions- Every function can performs a specific tast whenever we click on button.

#Function-1 Loading Dataset into application.

def loadDataset():    
    global filename
    global dataset
    outputarea.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    outputarea.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename)
    outputarea.insert(END,str(dataset.head()))

#Function-2 Data Preprocessing-Checking is there any NAN values exists in the dataset if not we can continue else we have to remove those rows form the dataset.  

def preprocessDataset():
    global x, y
    global dataset
    global X_train, X_test, y_train, y_test
    outputarea.delete('1.0', END)
    ## Checking missing entries in the dataset columnwise
    isna=dataset.isna().sum()
    outputarea.insert(END,str(isna))
    outputarea.insert(END,"\n\n"+str(dataset.isna()))
    y = dataset['label']
    x = dataset.drop(columns = ['label'])
    classes = {
               0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
               1:('bcc' , ' basal cell carcinoma'),
               2 :('bkl', 'benign keratosis-like lesions'),
               3: ('df', 'dermatofibroma'),
               4: ('nv', ' melanocytic nevi'),
               5: ('vasc', ' Vascular lesions'),
               6: ('mel', 'melanoma')
              }

#Function-3 Data Agumentation
'''
Data augmentation: Data Agumentation is a set of techniques to artificially increase the
                   amount of data by generating new data points from existing data.
                   This includes making small changes to data or using deep learning models
                   to generate new data points.
'''

def DataAgumentation():
      outputarea.delete('1.0', END)
      tabular_data = pd.read_csv('./Dataset/HAM10000_metadata.csv')
      #outputarea.tag_configure("center", justify='center')
      outputarea.insert(END,"\n\n"+str(tabular_data.head()))

#Function-4 Graph to display Frequency Distribution of Classes

def FreqDC():
      outputarea.delete('1.0', END)
      filename='./Dataset/HAM10000_metadata.csv'
      tabular_data = pd.read_csv(filename)
      sns.countplot(x = 'dx', data = tabular_data)
      plt.xlabel('Disease', size=12)
      plt.ylabel('Frequency', size=12)
      plt.title('Frequency Distribution of Classes', size=16)
      plt.show()
      #outputarea.tag_configure("center", justify='center')
      outputarea.insert(END,"\n\nbkl-Benign keratosis-like lesions\n\n"+"nv-Melanocytic Nevi\n\n"+"df-Dermatofibroma\n\n"+"mel-Melanoma\n\n"+"vasc-ascular lesions\n\n"+"bcc-Basal Cell Carcinoma\n\n"+"akiec-Actinic Keratoses and Inteaepithelial carcinomae\n\n")
      

#Function-5 Graph to display Age vs Count

def AgevsCount():
      filename='./Dataset/HAM10000_metadata.csv'
      tabular_data = pd.read_csv(filename)
      outputarea.delete('1.0', END)
      bar, ax = plt.subplots(figsize=(10,10))
      sns.histplot(tabular_data['age'])
      plt.title('Histogram of Age of Patients', size=16)
      value = tabular_data[['localization', 'sex']].value_counts().to_frame()
      value.reset_index(level=[1,0 ], inplace=True)
      temp = value.rename(columns = {'localization':'location', 0: 'count'})
      bar, ax = plt.subplots(figsize = (12, 12))
      sns.barplot(x = 'location',  y='count', hue = 'sex', data = temp)
      plt.title('Location of disease over Gender', size = 16)
      plt.xlabel('Disease', size=12)
      plt.ylabel('Frequency/Count', size=12)
      plt.xticks(rotation = 90)
      plt.show()

#Function-6 Random oversampling
'''Which involves randomly duplicating examples from the minority class
   and adding them to the training dataset.'''

def RunRandOverSamp():
      global x,y
      outputarea.delete('1.0', END)
      oversample = RandomOverSampler()
      x,y  = oversample.fit_resample(x,y)
      x = np.array(x).reshape(-1,28,28,3)
      print('Shape of X :',x.shape)
      outputarea.insert(END,'Shape of X :\t'+str(x.shape))

#Function-7 Generating Training and Testing splits

def Split():
      global X_train, X_test, Y_train, Y_test
      outputarea.delete('1.0', END)
      #x = (x-np.mean(x))/np.std(x)
      X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.2, random_state=1)
      outputarea.insert(END,"\n\nDataset Length : "+str(len(x))+"\n")
      outputarea.insert(END,"Total length used for training : "+str(len(X_train))+"\n")
      outputarea.insert(END,"Total length used for testing  : "+str(len(X_test))+"\n")

#Function-8 Generating CNN model

def CNN():
      global model
      outputarea.delete('1.0', END)
      model = Sequential()
      model.add(Conv2D(16, kernel_size = (3,3), input_shape = (28, 28, 3), activation = 'relu', padding = 'same'))
      model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu'))
      model.add(MaxPool2D(pool_size = (2,2)))
      model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = 'same'))
      model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
      model.add(MaxPool2D(pool_size = (2,2), padding = 'same'))
      model.add(Flatten())
      model.add(Dense(64, activation='relu'))
      model.add(Dense(32, activation='relu'))
      model.add(Dense(7, activation='softmax'))
      msg=model.summary()
      callback = tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_acc', mode='max', verbose=1)
#Initial Training
      model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
      history = model.fit(X_train, Y_train, validation_split=0.2, batch_size = 128, epochs = 20, callbacks=[callback])
      outputarea.insert(END,"Finally the Model is Loaded with 20 epochs")

#Function-9 Generating Accuracy graph

def Acc():
      outputarea.delete('1.0', END)
      loss, acc = model.evaluate(X_test, Y_test, verbose=2)
      acc  *= 100
      print("Generated Accuracy = ",ceil(acc))
      outputarea.insert(END,"Accuracy of the Model is : "+str(ceil(acc)))
      #AccGraph()
      
def AccGraph():
      plt.plot(history.history['accuracy'])
      plt.plot(history.history['val_accuracy'])
      plt.title('model accuracy')
      plt.ylabel('accuracy')
      plt.xlabel('epoch')
      plt.legend(['train', 'val'], loc='upper left')
      plt.show()

#Function-10  For Closing the Desktop Application finally! Happy Ending...!!!

def close():
    main.destroy()

#Creating Lables, Buttons and TextArea using Tkinter module.

import tkinter.font as font

bold_font = font.Font(weight="bold")
font = ('times', 25, 'bold')
title = Label(main, text='Skin Disease Diagnosis Using Convolutional Neural Network', font=bold_font)
title.config(bg='Black', fg='gold')  
title.config(font=font)           
title.config(height=3, width=68)       
title.place(x=0,y=0)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload HAM10000 Dataset", command=loadDataset , padx=5, font=bold_font,
    pady=5, bg='#4a7abc', fg='white', activebackground='green', activeforeground='white')
uploadButton.place(x=120,y=160)
uploadButton.config(font=ff)
uploadButton.config(width=30)

processButton = Button(main, text="Preprocess Dataset", command=preprocessDataset, padx=5, font=bold_font,
    pady=5, bg='#4a7abc', fg='white', activebackground='green', activeforeground='white')
processButton.place(x=120,y=220)
processButton.config(font=ff)
processButton.config(width=30)

DAButton = Button(main, text="Data Agumentation", command=DataAgumentation , padx=5, font=bold_font,
    pady=5, bg='#4a7abc', fg='white', activebackground='green', activeforeground='white')
DAButton.place(x=120,y=280)
DAButton.config(font=ff)
DAButton.config(width=30)

graph1Button = Button(main, text="Frequency Distribution Count", command=FreqDC , padx=5, font=bold_font,
    pady=5, bg='#4a7abc', fg='white', activebackground='green', activeforeground='white')
graph1Button.place(x=120,y=340)
graph1Button.config(font=ff)
graph1Button.config(width=30)

AvCButton = Button(main, text="Plot Age vs Count", command=AgevsCount ,padx=5, font=bold_font,
    pady=5, bg='#4a7abc', fg='white', activebackground='green', activeforeground='white')
AvCButton.place(x=120,y=400)
AvCButton.config(font=ff)
AvCButton.config(width=30)

RROverSampleButton = Button(main, text="Run Random OverSampling", command=RunRandOverSamp , padx=5, font=bold_font,
    pady=5, bg='#4a7abc', fg='white', activebackground='green', activeforeground='white')
RROverSampleButton.place(x=120,y=460)
RROverSampleButton.config(font=ff)
RROverSampleButton.config(width=30)

SplitButton = Button(main, text="Spliting Dataset", command=Split , padx=5, font=bold_font,
    pady=5, bg='#4a7abc', fg='white', activebackground='green', activeforeground='white')
SplitButton.place(x=120,y=520)
SplitButton.config(font=ff)
SplitButton.config(width=30)

CNNButton=Button(main, text="Run CNN",command=CNN , padx=5,font=bold_font,
    pady=5, bg='#4a7abc', fg='white', activebackground='green', activeforeground='white')
CNNButton.place(x=120,y=580)
CNNButton.config(font=ff)
CNNButton.config(width=30)

AccButton=Button(main, text="Accuracy Generation",command=Acc , padx=5, font=bold_font,
    pady=5, bg='#4a7abc', fg='white', activebackground='green', activeforeground='white')
AccButton.place(x=120,y=640)
AccButton.config(font=ff)
AccButton.config(width=30)

exitButton = Button(main, text="Logout", command=close ,padx=5, font=bold_font,
    pady=5, bg='#4a7abc', fg='white', activebackground='green', activeforeground='white')
exitButton.place(x=120,y=700)
exitButton.config(font=ff)
exitButton.config(width=30)

font1 = ('times', 12, 'bold')
outputarea = Text(main,height=31,width=89)
scroll = Scrollbar(outputarea)
outputarea.configure(yscrollcommand=scroll.set)
outputarea.place(x=470,y=150)
outputarea.config(font=font1)

main.config()
main.mainloop()
