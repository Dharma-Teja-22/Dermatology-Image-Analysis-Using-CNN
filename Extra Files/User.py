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
from PIL import Image, ImageTk
import webbrowser 

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

exitButton = Button(main, text="Logout", command=close ,padx=5, font=bold_font,
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
