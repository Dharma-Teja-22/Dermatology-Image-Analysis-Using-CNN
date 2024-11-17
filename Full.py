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
import webbrowser
import tkinter.font as font

main = tkinter.Tk()
main.title("Skin Disease Diagnosis Using ConvNN!")
main.geometry("1300x1200")

global password_login_entry
global username_login_entry
global username_verify
global password_verify
username_verify = StringVar()
password_verify = StringVar()

def AdminLogin():
      global password_login_entry
      global username_login_entry
      username = username_verify.get()
      password = password_verify.get()
      username_login_entry.delete(0, END)
      password_login_entry.delete(0, END)    
      if username == 'Admin' and password == 'Admin':
            AdminInterface()
      else:
            messagebox.showinfo("Invalid Login Details","Invalid Login Details")       

def AdminInterface():
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
                  amount of data by generating new data points from existing data. This includes 
                  making small changes to data or using deep learning models to generate new data points.
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


def UserInterface():
      main = Tk()
      main.title("Skin Disease Diagnosis Using Convolutional Neural Network")
      main.geometry("1300x1200")
      #main.config(bg="powder blue")
      main.config(bg="LightBlue1")

      global check
      #Function-1 Image uploading for classification task and display it in the textarea

      def UploadImg():
            outputarea.delete('1.0', END)
            global filename, image
            outputarea.delete('1.0', END)
            filename = filedialog.askopenfilename(initialdir="Dataset/HAM10000_images_part_1", filetypes=[("Image Files", "*.png *.jpg")])
            outputarea.insert(END,"The Selected image path : \n\n"+filename)
            image = cv2.imread(filename)
            img=cv2.resize(image, (320,280))
            cv2.imshow("Selected Image", img)
      
      #Function-2 Generating Which type of Disease for the given images is to be generated using this function.

      def GeneDisease():
            global pred
            outputarea.delete('1.0', END)
            img=filename
            image=Image.open(img)
            image=np.array(image)
            image=cv2.resize(image,(28,28))/255
            image=np.expand_dims(image,axis=0)
            base=os.getcwd()
            path=os.path.join(os.path.join(base,'home'),'static_home')
            path=os.path.join(path,'best_model.h5')
            model=load_model(path)
            prediction = model.predict(image)
            pred=np.argmax(prediction)
            label = {
                  0:'Actinic keratoses',
                  1:'Basal cell carcinoma',
                  2:'Seborrhoeic Keratosis',
                  3:'Dermatofibroma',
                  4:'Melanocytic nevi',
                  5:'Vascular lesions',
                  6:'Melanoma'
            }
            #print("The output num of predicted disease = ",pred)
            if pred>=0 and pred<7:
                  disease=label[pred]
                  outputarea.insert(END,"The Uploaded Microscopic image contains a disease is : \n\n"+disease)
                  #cv2.putText(image, f"Disease = {disease}", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                  #cv2.putText(image, f"Disease = {disease}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9)
                  image=cv2.imread(filename)
                  Gimage = cv2.putText(image, f'{disease}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,100,0), 4, cv2.LINE_AA) 
                  img=cv2.resize(Gimage, (320,280))
                  cv2.imshow("Generated Disease Image",img)
            else:
                  outputarea.insert(END,"Choose correct Microscopic images to get result")

      #Function-3 Information regarding the disease.
      def Info():
            global check
            outputarea.delete('1.0', END)
            if pred==0:
                  check=1
                  outputarea.insert(END,"\t\t\t  Actinic keratoses\n\n\n"+"An actinic keratosis develops in areas of skin that have undergone repeated or long-term exposure to the sun's UV light, and it is a warning sign of increased risk of skin cancer. About 10% to 15% of actinic keratoses eventually change into squamous cell cancers of the skin. It is a small bump that feels like sandpaper or a small, scaly patch of sun-damaged skin that has a pink, red, yellow or brownish tint."+"It's also known as a solar keratosis\n\n"+"Prevention:\n\n"+"Limit your time in the sun.\nCover up\nUse sunscreen\nAvoid tanning beds\nCheck your skin regularly and report changes to your health care provider.\n\n")
            elif pred==1:
                  check=1
                  outputarea.insert(END,"\t\t\t\tBasal cell carcinoma\n\n\n"+"One of three main types of cells in the top layer of the skin, basal cells shed as new ones form. BCC most often occurs when DNA damage from exposure to ultraviolet (UV) radiation from the sun or indoor tanning triggers changes in basal cells in the outermost layer of skin (epidermis), resulting in uncontrolled growth.\n\n"+"It’s important to note that BCCs can look quite different from one person to another. \n\n")
            elif pred==2:
                  check=1
                  outputarea.insert(END,"\t\t\t \t Seborrhoeic Keratosis\n\n"+"Experts don't completely understand what causes a seborrheic keratosis. This type of skin growth does tend to run in families, so there is likely an inherited tendency.\n If you've had one seborrheic keratosis, you're at risk of developing others.\n\n"+"A seborrheic keratosis is a non-cancerous (benign) skin tumour that originates from cells, namely keratinocytes, in the outer layer of the skin called the epidermis. Like liver spots, seborrheic keratoses are seen more often as people age\n\n")
            elif pred==3:
                  check=1
                  outputarea.insert(END,"\t\t\t\t Dermatofibroma\n\n"+"Dermatofibromas are small, harmless growths that appear on the skin. These growths, or papules, can develop anywhere on the body, but they are most common on the arms, lower legs, and upper back.\n\n"+"Small Injury or bug bites in the area where the lesion later forms may contirbutes to this type of development.\n\n")
            elif pred==4:
                  check=1
                  outputarea.insert(END,"\t\t\t\t Melanocytic nevi\n\n"+"Melanocytic nevi commonly known as moles.\n\n"+"It is thought to result from a defect in embryologic development during the first twelve weeks of pregnancy.\n\n"+"Certainly! Melanocytic nevi, commonly known as moles, are noncancerous skin conditions involving pigment-producing cells.\n\n")
            elif pred==5:
                  check=1
                  outputarea.insert(END,"\t\t\t\t  Vascular lesions\n\n"+"Vascular lesions are relatively common abnormalities of the skin and underlying tissues, more commonly known as birthmarks. There are three major categories of vascular lesions: Hemangiomas, Vascular Malformations, and Pyogenic Granulomas. While these birthmarks can look similar at times, they each vary in terms of origin and necessary treatment.\n\n"+"Injury to the vessel walls byb bacteria or by virus.\n\nOccures at birthmarks too.\n\n")
            elif pred==6:
                  check=1
                  outputarea.insert(END,"\t\t\t\t   Melanoma\n\n"+"Look for anything new, changing or unusual on both sun-exposed and sun-protected areas of the body. Melanomas commonly appear on the legs of women. The number one place they develop on men is the trunk. Keep in mind, though, that melanomas can arise anywhere on the skin, even in areas where the sun doesn’t shine.\n\n"+"65% of this type of skin disease is occured due to sun exposure (or) by UV light.\n\n")


      #Function-4 For opening a New tab for displaying hospital information nearer to you!

      def NearbyHos():
            if check==1:
                  webbrowser.open("https://www.google.com/search?q=dermatologist+near+me")
            else:
                  outputarea.insert(END,"\n\n Insert Any Image and do processing again after processing we can display nearby hospital information!")

      #Function-5  For Closing the Desktop Application finally! Happy Ending...!!!

      def close():
            main.destroy()

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

      UploadImg=Button(main,text="Upload histopathology Skin Image",command=UploadImg ,padx=5, font=bold_font,
      pady=5,bg='#4a7abc', fg='white', activebackground='green', activeforeground='white')
      UploadImg.place(x=120,y=250)
      UploadImg.config(font=ff)
      UploadImg.config(width=30)

      GenDis=Button(main,text="Generate Disease for given  Skin Image",command=GeneDisease , padx=5, font=bold_font ,
      pady=5, bg='#4a7abc', fg='white', activebackground='green', activeforeground='white')
      GenDis.place(x=120,y=320)
      GenDis.config(font=ff)
      GenDis.config(width=30)

      Info=Button(main,text="Information about given Image",command=Info ,padx=5, font=bold_font,
      pady=5, bg='#4a7abc', fg='white', activebackground='green', activeforeground='white')
      Info.place(x=120,y=390)
      Info.config(font=ff)
      Info.config(width=30)

      hospital=Button(main,text="NearBy Hospital Information",command=NearbyHos , padx=5, font=bold_font,
      pady=5, bg='#4a7abc', fg='white', activebackground='green', activeforeground='white')
      hospital.place(x=120,y=460)
      hospital.config(font=ff)
      hospital.config(width=30)

      exitButton = Button(main, text="Close", command=close ,padx=5, font=bold_font,
      pady=5, bg='#4a7abc', fg='white', activebackground='green', activeforeground='white')
      exitButton.place(x=120,y=530)
      exitButton.config(font=ff)
      exitButton.config(width=30)

      font1 = ('times', 12, 'bold')
      outputarea = Text(main,height=28,width=89)
      scroll = Scrollbar(outputarea)
      outputarea.configure(yscrollcommand=scroll.set)
      outputarea.place(x=490,y=150)
      outputarea.config(font=font1)

      main.config()
      main.mainloop()

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

font1 = ('times', 20, 'bold')

l = Label(main, text="  About Project  ", font=bold_font, bg="yellow")
l.place(relx=0.5, rely=0.19, anchor="center")
l.config(font=font1)

import tkinter.font as font
font2 = ('times', 14, 'bold')
label_font = font.Font(slant="italic")
l0 = Label(main, text="ADMIN can perform operations like  - Upload Dataset, Preprocess, Data Augmentation, Plotting Graphs, Run RandomOverSampler, Splitting dataset into training and testing sets, Run ConvNN, Generating Accuracy.\n", wraplength=1300, justify="left", font=label_font)
l0.place(x=10,y=200)
l0.config(font=font2,  bg="skyblue")  


l1 = Label(main, text="USER can perform operations like   - Upload any histopathology image, Generate Disease, Information about the generated disease, Nearby Hospital Info, close.\n", wraplength=1300, justify="left", font=label_font)
l1.place(x=10,y=250)
l1.config(font=font2, bg="skyblue")  

font = ('times', 20, 'bold')
l2 = Label(main, text="  Admin Login  ", font=bold_font, bg="yellow")
l2.place(relx=0.5, rely=0.39, anchor="center")
l2.config(font=font)

font1 = ('times', 15, 'bold')
l3 = Label(main, text="Username * ")
l3.place(x=500,y=380)
l3.config(font=font1)  
username_login_entry = Entry(main, textvariable=username_verify)
username_login_entry.place(x=620,y=380, width=150, height=30)

font1 = ('times', 15, 'bold')
l4 = Label(main, text="Password  *")
l4.place(x=500,y=440)
l4.config(font=font1)  
password_login_entry = Entry(main, textvariable=password_verify, show= '*')
password_login_entry.place(x=620,y=440, width=150, height=30)

Admin = Button(main, text="Admin Login", command=AdminLogin, padx=5, font=bold_font, pady=5, bg='#4a7abc', fg='white', activebackground='green', activeforeground='white')
Admin.place(relx=0.5, rely=0.6, anchor="center")
Admin.config(font=font1)
Admin.config(width=30)  

font = ('times', 20, 'bold')
l5 = Label(main, text="  User Login  ", font=bold_font, bg="yellow")
l5.place(relx=0.5, rely=0.69, anchor="center")
l5.config(font=font)

l5 = Label(main, text="Click on below  button to access the user interface", font=bold_font, bg="skyblue")
l5.place(relx=0.5, rely=0.74, anchor="center")
l5.config(font=font1)

User = Button(main, text="User View", command=UserInterface, padx=5, font=bold_font, pady=5, bg='#4a7abc', fg='white', activebackground='green', activeforeground='white')
User.place(relx=0.5, rely=0.8, anchor="center")
User.config(font=font1)
User.config(width=30)

exitButton = Button(main, text="Close", command=close ,padx=5, font=bold_font, pady=5, bg='#4a7abc', fg='white', activebackground='green', activeforeground='white')
exitButton.place(relx=0.93, rely=0.87)
exitButton.config(font=font1)

main.config(bg='skyblue')
main.mainloop()