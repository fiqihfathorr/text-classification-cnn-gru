import tkinter
from tkinter import *
from tkinter import filedialog as fd
from pandastable import Table
import tkinter.messagebox
from tksheet import Sheet
import customtkinter
from PIL import Image, ImageTk
import os
from os import path
import numpy as np
import pickle as pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import threading

PATH = os.path.dirname(os.path.realpath(__file__))

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

def onehotencoding(data,max_lenght):
    """ 
    Tujuan : Fungsi untuk pembentukan input matriks
    Input : data-> 1 data teks ulasan, max_lenght-> Jumlah maksimal karakter teks ulasan   
    Output : onehot_encoded -> representasi matriks dari teks ulasan
    """
    #list Karakter-karakter yang direpresentasikan (38 karakter)
    alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789 \n'
    #Membuat kamus karakter menjadi nilai numerik
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    #mengubah data menjadi nilai numerik
    integer_encoded = [char_to_int[char] for char in data]
    #Proses onehotencoding
    onehot_encoded = []
    #Untuk setiap nilai pada interger_encoded
    for nilai in integer_encoded:
        #membuat list bernilai 0 berukuran panjang alphabet
        letter = [0 for i in range(len(alphabet))]
        #mengubah nilai pada letter menjadi 1 untuk data letter ke-nilai
        letter[nilai] = 1
        onehot_encoded.append(letter)
    while len(onehot_encoded)< max_lenght :
        letter = [0 for i in range(len(alphabet))]
        onehot_encoded.append(letter)
    return onehot_encoded

#membuat class untuk CNN
class convolutional:
    #inisialisasi awal
    def __init__(self,n_filter,y_filter,n_feature,bobot=[]):
        """ 
        Tujuan : Inisialisasi parameter pada CNN
        Input : n_filter -> banyak kernel, y_filter -> ukuran baris kernel, n_feature -> banyak feature map karakter 
        """
        self.n_filter=n_filter
        self.y_filter=y_filter
        self.n_feature=n_feature
        #membangkitkan kernel dengan He weight Initialization
        if len(bobot)==0:
            self.filters=np.random.randn(n_filter,y_filter,n_feature)*np.sqrt(2/(y_filter*n_feature))
        else:
            self.n_filter=n_filter
            self.y_filter=y_filter
            self.n_feature=n_feature
            self.filters=bobot
            
    def konvolusi(self, data):
        """
        Tujuan : Prosedur konvolusi
        Input : data -> input matriks dari representasi teks ulasan 
        Output : hasil -> list hasil operasi konvolusi / feature map
        """
        hasil=[]
        for h in range(self.n_filter):
            output=[]
            for i in range(len(data)-len(self.filters[h])+1):
                temp=0
                for j in range(len(self.filters[h])):
                    for k in range(len(self.filters[h][0])):
                        temp+=data[j+i][k]*self.filters[h][j][k]
                output.append(temp)
            hasil.append(output)
        return np.array(hasil)
    
    def relu(self,x):
        """
        Tujuan : Proses mengubah nilai hasil konvolusi dengan fungsi aktivasi Relu
        Input : x -> list hasil konvolusi 
        Output: output -> list hasil konvolusi hasil relu 
        """
        output=[]
        for kernel in x:
            wadah=[]
            #if nilai<0 then nilai=0, else nilai=nilai
            for value in kernel:
                if value < 0:
                    wadah.append(0)
                else:
                    wadah.append(value)
            output.append(wadah)
        return np.array(output)
    
    def pooling(self,x):
        """
        Tujuan : Prosedur pooling CNN
        Input : x -> list hasil relu
        output : output -> list hasil pooling
        """
        output=[]
        #ambil data pada setiap hasil operasi kernel
        for i in range(len(x)):
            #mencari nilai maks untuk pada setiap feature map
            wadah=x[i][0]
            for j in range(len(x[i])-1):
                if x[i][j]>wadah:
                    wadah=x[i][j]
            output.append([wadah])
        return np.array(output)

def lowercase(data):
    '''
    TujuanFungsi digunakan untuk menstandarkan semua teks kedalam huruf kecil
    Input : data-> string
    '''
    return data.lower()

def cleantext(text):
    '''Untuk menghilangkan karakter yang tidak diinginkan dalam teks
    argumen:
        text-> string
    '''
    used_char=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','1','2','3','4','5','6','7','8','9','0',' ']
    for x in text:
        if x not in used_char:
            text=text.replace(x,'')
    return text.encode('ascii', 'ignore').decode('ascii')

def sigmoid(x):
    '''
    Tujuan :fungsi aktivasi sigmoid
    Input : x -> list
    Output: hasil -> hasil fungsi aktivasi sigmoid [0,1]
    '''
    hasil=[]
    for nilai in x:
        if nilai>=0 :
            z=np.exp(-nilai)
            sig=1/(1+z)
        else:
            z=np.exp(nilai)
            sig=z/(1+z)
        hasil.append(sig)
    hasil=np.array(hasil)
    return hasil

def tanh(x):
    '''
    Tujuan : Fungsi aktivasi tanh
    Input : x -> list
    Output : hasil -> hasil fungsi aktivasi tanh [-1,1]
    '''
    return (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))

def softmax(vector):
    '''
    Tujuan : Fungsi aktivasi softmax
    Input : vector -> List
    Output : hasil -> list hasil softmax
    '''  
    e = np.exp(vector)
    return e / e.sum()

def maxmin(a,kernel):
    """
    Tujuan : Mencari nilai maksimal dan minimal dari data
    Input : a -> matriks data numerik
            kernel -> ukuran kernel
    Output : maxmin -> list maks dan min untuk setiap kolom data
    """
    maxmin=[]
    for i in range(len(a[0])):
        max_value=a[0][i][0]
        min_value=a[0][i][0]
        for j in range(len(a)):
            if a[j][i]>max_value:
                max_value=a[j][i][0]
            if a[j][i]<min_value:
                min_value=a[j][i][0]
        maxmin.append([max_value,min_value])
    #menyimpan hasil maxmin data
    with open(PATH+"/hasil/"+str(kernel)+"/maxmindata.txt","wb") as file:
        pickle.dump(maxmin,file)
    file.close()
    return maxmin

def normalisasi(a,maxmin):
    """
    Tujuan :menormalisasi data
    Input : a-> data, maxmin-> data maks dan min dari data
    Output : a-> hasil normalisasi data
    """
    for row in a:
        for i in range(len(row)):
            row[i]=(row[i]-maxmin[i][1])/(maxmin[i][0]-maxmin[i][1])
    return a

class GRU():
    def __init__(self, n_input, n_hidden, n_output,bobot_optimal=None):
        """
        Tujuan : Inisialisasi parameter GRU
        Input : n_input -> jumlah data input
                n_hidden -> jumlah hidden node
                n_output -> jumlah node output
                bobot_optimal -> inisilasisasi bobot dan bias optimal (jika sudah ada)
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.w={}
        self.b={}
        
        if bobot_optimal is None:
            #inisialisasi bobot reset gate
            self.w['wr'] = np.random.randn(n_hidden, n_input)*np.sqrt(2/(n_input+n_output))
            self.b['r'] = np.zeros((n_hidden, 1))
            self.w['ur'] = np.random.randn(n_hidden, n_hidden)*np.sqrt(2/(n_input+n_output))
            #insisialisasi bobot update gate
            self.w['wz'] = np.random.randn(n_hidden, n_input)*np.sqrt(2/(n_input+n_output))
            self.b['z'] = np.zeros((n_hidden, 1))
            self.w['uz'] = np.random.rand(n_hidden, n_hidden)*np.sqrt(2/(n_input+n_output))
            #inisialisasi bobot final memory
            self.w['w_h'] = np.random.randn(n_hidden, n_input)*np.sqrt(2/(n_input+n_output))
            self.b['_h'] = np.zeros((n_hidden, 1))
            self.w['u_h'] = np.random.randn(n_hidden, n_hidden)*np.sqrt(2/(n_input+n_output))
            #inisiali bobot output
            self.w['wo'] = np.random.randn(n_output, n_hidden)*np.sqrt(2/(n_input+n_output))
            self.b['o'] = np.zeros((n_output, 1))
            self.h_s = np.zeros((n_hidden,1))
            self.h = np.zeros((n_hidden,1))
        else :
            #inisialisasi bobot reset gate
            self.w['wr'] = bobot_optimal[0]['wr']
            self.b['r'] = bobot_optimal[1]['r']
            self.w['ur'] = bobot_optimal[0]['ur']
            #insisialisasi bobot update gate
            self.w['wz'] = bobot_optimal[0]['wz']
            self.b['z'] = bobot_optimal[1]['z']
            self.w['uz'] = bobot_optimal[0]['uz']
            #inisialisasi bobot final memory
            self.w['w_h'] = bobot_optimal[0]['w_h']
            self.b['_h'] = bobot_optimal[1]['_h']
            self.w['u_h'] = bobot_optimal[0]['u_h']
            #inisiali bobot output
            self.w['wo'] = bobot_optimal[0]['wo']
            self.b['o'] = bobot_optimal[1]['o']
            self.h_s = np.zeros((n_hidden,1))
            self.h = np.zeros((n_hidden,1))        
        
    def forward(self, inputs):
        """
        Tujuan : Umpan maju GRU
        Input : inputs -> Matriks input dari proses CNN
        Output : o -> list output
        """
        self.inputs=inputs
        # calculating reset gate value
        self.r=sigmoid(np.dot(self.w['wr'],inputs) + np.dot(self.w['ur'], self.h_s) + self.b['r'])
        # print("r = ",self.r)
        #print("ukuran r = ",np.shape(self.r))
        # calculation update gate value
        self.z=sigmoid(np.dot(self.w['wz'],inputs) + np.dot(self.w['uz'], self.h_s)  + self.b['z'])
        #print("z = ",self.z)
        #print("ukuran z = ",np.shape(self.z))
        # applying candidate final memory
        self._h=tanh(np.dot(self.w['w_h'], inputs) + np.multiply(self.r, np.dot(self.w['u_h'], self.h_s)) + self.b['_h'])
        #print("_h = ",self._h)
        #print("ukuran _h = ",np.shape(self._h))
        # applying final memory
        self.h = np.multiply(self.z, self.h_s) + np.multiply(1-self.z, self._h)
        #print("h = ",self.h)
        #print("ukuran h = ",np.shape(self.h))
        # calculating output
        self.o=softmax(np.dot(self.w['wo'], self.h) + self.b['o'])
        #print("o = ",self.o)
        #print("ukuran o = ",np.shape(self.o))
        return self.o
    
    def cross_entropy(self,y):
        """
        Tujuan : menghitung cross-entropy error
        Input : y -> target, o -> prediksi
        Ouput : loss -> nilai cross-entropy error
        """
        loss=0
        for i in range(len(y)):
            loss+=y[i]*np.log(self.o[i]+0.000001)
        #mean=1/len(self.o)*loss
        return -loss

    def backward(self,target):
        """
        Tujuan : Umpan mundur GRU
        Input : o -> output
                target -> target
        Output : dw -> dE/dw untuk masing-masing bobot
                db -> dE/db untuk masing-masing bias
        """
        self.target=target
        self.E = self.cross_entropy(target)
        self.dw={}
        self.db={}
        self.dw['wz'] = np.zeros((self.n_hidden, self.n_input))
        self.db['z'] = np.zeros((self.n_hidden, 1))
        self.dw['uz'] = np.zeros((self.n_hidden, self.n_hidden))

        # reset dw
        self.dw['wr'] = np.zeros((self.n_hidden, self.n_input))
        self.db['r'] = np.zeros((self.n_hidden, 1))
        self.dw['ur'] = np.zeros((self.n_hidden, self.n_hidden))

        # _h dw
        self.dw['w_h'] = np.zeros((self.n_hidden, self.n_input))
        self.db['_h'] = np.zeros((self.n_hidden, 1))
        self.dw['u_h'] = np.zeros((self.n_hidden, self.n_hidden))

        # hidden-2-output dw
        self.dw['wo'] = np.zeros((self.n_output, self.n_hidden))
        self.db['o'] = np.zeros((self.n_output, 1))
        
        # gradient at output layer
        go = self.o - self.target
            
        # update dw pada output layer
        self.dw['wo'] = np.dot(go, self.h.T)
        self.db['o'] = go

        dh = np.dot(self.w['wo'].T,go)
        d_h = np.multiply(dh,(1-self.z))
        d_h_ = d_h*(1- self._h**2)
        
        # gradient at hidden layer
        dz = np.multiply(dh,self.h_s-d_h)
        dz_ = dz * (self.z*(1-self.z))
                
        temp = np.dot(self.w['u_h'].T, d_h_)
        dr = np.multiply(temp,self.h_s)
        dr_ = dr*(self.r*(1-self.r))
        
        # calculating reset dw
        self.dw['wr'] = np.dot(dr_ , self.inputs.T)
        self.db['r'] = dr_
        self.dw['ur'] = np.dot(dr_ , self.h_s.T)

        # calculating update dw
        self.dw['wz'] = np.dot(dz_, self.inputs.T)
        self.db['z'] = dz_
        self.dw['uz'] = np.dot(dz_, self.h_s.T)
        
        self.dw['w_h'] = np.dot(d_h_, self.inputs.T)
        self.db['_h'] = d_h_
        self.dw['u_h'] = np.dot(d_h_, np.multiply(self.r, self.h_s).T)
        # print("dw :",self.dw)
        # print("db : ",self.db)
        self.h_s=self.h
    
    def update(self,alpha):
        """
        Tujuan : Menghitung bobot dan bias baru
        Input : alpha -> learning rate, w -> bobot, b-> bias
        Output : w -> bobot baru, b -> bias baru
        """
        self.w['wo']-=alpha*self.dw['wo']
        self.b['o']-=alpha*self.db['o']
        self.w['wr']-=alpha*self.dw['wr']
        self.w['ur']-=alpha*self.dw['ur']
        self.b['r']-=alpha*self.db['r']
        self.w['w_h']-=alpha*self.dw['w_h']
        self.w['u_h']-=alpha*self.dw['u_h']
        self.b['_h']-=alpha*self.db['_h']
        self.w['wz']-=alpha*self.dw['wz']
        self.w['uz']-=alpha*self.dw['uz']
        self.b['z']-=alpha*self.db['z']
    
    def train(self,inputs,target,alpha,max_epoch,min_error):
        """
        Tujuan : Pelatihan data dengan GRU
        Input : inputs -> data input setelah CNN
                target -> list target
                alpha -> learning rate
                max_epoch -> jumlah iterasi
                min_error -> nilai error minimal yang diharapkan
        Output : w -> bobot optimal, b-> bias optimal, loss-> Error pelatihan
        """
        epoch=1;loss=9999999999999999
        global eror,iterasi
        eror=[]
        iterasi=[]
        alpha0=alpha
        while epoch<=max_epoch and loss>=min_error:
            loss=0
            for x,y in zip(inputs,target):
                #print("data",x)
                self.forward(x)
                self.backward(y)
                self.update(alpha)
                #print("bobot :", self.w)
                #print("bias :",self.b)
                loss+=self.E
            #if epoch%10==0 or epoch==1 or epoch==max_epoch:
            temp_data=[epoch,alpha,loss[0]/len(inputs)]
            label_info_1.insert('',tkinter.END, values=temp_data)
            #if epoch%10==0 :
            #    alpha=alpha0*(1./(1.+0.2*epoch))
            eror.append(loss/len(inputs))
            iterasi.append(epoch)
            epoch=epoch+1
        return

class App(customtkinter.CTk):

    WIDTH = 900
    HEIGHT = 640

    def __init__(self):
        super().__init__()

        self.title("Aplikasi Klasifikasi Teks Ulasan dengan Hybrid Convolutional Neural Network dan Gated Recurrent Unit")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        # self.minsize(App.WIDTH, App.HEIGHT)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # call .on_closing() when app gets closed

        # ============ create two frames ============

        # configure grid layout (2x1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame_left = customtkinter.CTkFrame(master=self,
                                                 width=180,
                                                 corner_radius=0)
        self.frame_left.grid(row=0, column=0, sticky="nswe")

        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=1, sticky="nswe", padx=20, pady=20)

        # ============ frame_left ============

        # configure grid layout (1x11)
        self.frame_left.grid_rowconfigure(0, minsize=10)   # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(5, weight=1)  # empty row as spacing
        self.frame_left.grid_rowconfigure(8, minsize=20)    # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(11, minsize=10)  # empty row with minsize as spacing

        self.unair_image = ImageTk.PhotoImage(Image.open(PATH + "/image/AIRLANGGA.png").resize((100, 100), Image.ANTIALIAS))
        self.unair_button = customtkinter.CTkButton(master=self.frame_left, image=self.unair_image, text="", width=50, height=50,
                                   corner_radius=10, fg_color="gray40", hover_color="gray25")
        self.unair_button.grid(row=1, column=0, columnspan=1, padx=10, pady=10)

        self.label_1 = customtkinter.CTkLabel(master=self.frame_left,
                                              text="Universitas Airlangga",
                                              )  # font name and size in px
        self.label_1.grid(row=2, column=0, pady=10, padx=10)

        self.aplikasi = customtkinter.CTkButton(master=self.frame_left,
                                                text="Aplikasi",
                                                fg_color=("gray75", "gray30"),  # <- custom tuple-color
                                                command=self.aplikasi_klasifikasi)
        self.aplikasi.grid(row=3, column=0, pady=10, padx=20)

        self.ekstraksi_cnn = customtkinter.CTkButton(master=self.frame_left,
                                                text="Ekstraksi CNN",
                                                fg_color=("gray75", "gray30"),  # <- custom tuple-color
                                                command=self.view_ekstraksi)
        self.ekstraksi_cnn.grid(row=4, column=0, pady=10, padx=20)

        self.pelatihan_gru = customtkinter.CTkButton(master=self.frame_left,
                                                text="Pelatihan GRU",
                                                fg_color=("gray75", "gray30"),  # <- custom tuple-color
                                                command=self.view_gru)
        self.pelatihan_gru.grid(row=5, column=0, pady=10, padx=20, sticky="n")

        self.developer = customtkinter.CTkButton(master=self.frame_left,
                                                text="Developer",
                                                fg_color=("gray75", "gray30"),  # <- custom tuple-color
                                                command=self.view_developer)
        self.developer.grid(row=6, column=0, pady=10, padx=20, columnspan=3, sticky="n")

        self.help = customtkinter.CTkButton(master=self.frame_left,
                                                text="Bantuan Aplikasi",
                                                fg_color=("gray75", "gray30"),  # <- custom tuple-color
                                                command=self.view_help)
        self.help.grid(row=7, column=0, pady=10, padx=20)

        self.switch_2 = customtkinter.CTkSwitch(master=self.frame_left,
                                                text="Dark Mode",
                                                command=self.change_mode)
        self.switch_2.grid(row=10, column=0, pady=10, padx=20, sticky="w")

        # ============ frame_right ============

        # configure grid layout (3x7)
        self.frame_right.rowconfigure((0, 1, 2, 3,4,5,6,7,8), weight=1)
        self.frame_right.rowconfigure(9, weight=5)
        self.frame_right.columnconfigure((0, 1), weight=1)
        self.frame_right.columnconfigure(2, weight=0)

        self.frame_info = customtkinter.CTkFrame(master=self.frame_right)
        self.frame_info.grid(row=0, column=0, columnspan=2, rowspan=4, pady=20, padx=20, sticky="nsew")

        # ============ frame_info ============

        # configure grid layout (1x1)
        self.frame_info.rowconfigure(0, weight=1)
        self.frame_info.columnconfigure(0, weight=1)
        
        # ============ frame_right ============

        # set default values
        self.unair_button.configure(state=tkinter.DISABLED)
        self.switch_2.select()
        self.view_developer()

    def hapus():
        self.sentimenArea.configure(state='normal')
        self.sentimenArea.delete(0, END)
        self.sentimenArea.configure(state='readonly')
        self.positifArea.configure(state='normal')
        self.positifArea.delete(0, END)
        self.positifArea.configure(state='readonly')
        self.negatifArea.configure(state='normal')
        self.negatifArea.delete(0, END)
        self.negatifArea.configure(state='readonly')
        self.textArea.delete('1.0',END)

    def prediksi(self):
        nama_file=self.tf_pilih_bobot.get()
        e=self.textArea.get()
        if len(e)>500:
            tkinter.messagebox.showinfo("Informasi", "Mohon maaf banyak karakter yang dapat diinput sebanyak 500")
            return
        for root, dirs, files in os.walk(PATH+'/hasil'):
            for name in files:
                if name == nama_file:
                    tujuan=os.path.abspath(os.path.join(root, name))
                    jmlh_hidden=tujuan.split('_')[-3]
                    directory_path = os.path.dirname(os.path.abspath(tujuan))
        with open(str(tujuan),"rb") as file_:
            bobot_optimal=pickle.load(file_)
        file_.close()
        for root, dirs, files in os.walk(str(directory_path)):
            for name in files:
                if name.split('_')[:2][0]=="bobot" and name.split('_')[:2][1]=="cnn":
                    with open(str(os.path.abspath(os.path.join(root, name))),"rb") as file_:
                        bobot_cnn=pickle.load(file_)
                    file_.close()
        with open(str(directory_path)+"/maxmindata.txt","rb") as file_:
            maxmin=pickle.load(file_)
        file_.close()
        e=self.textArea.get()
        e=lowercase(e)
        e=cleantext(e)
        text_vector=onehotencoding(e,500)
        cnn=convolutional(bobot_cnn.shape[0], bobot_cnn.shape[1], 38,bobot_cnn)
        temp=cnn.konvolusi(text_vector)
        temp2=cnn.relu(temp)
        vektor_input=cnn.pooling(temp2)
        text_vector_norm=normalisasi(vektor_input,maxmin)
        gru=GRU(len(text_vector_norm),int(jmlh_hidden),2,bobot_optimal)
        output=gru.forward(text_vector_norm)
        self.negatifArea.configure(state=tkinter.NORMAL)
        self.negatifArea.delete(0, tkinter.END)
        self.negatifArea.insert(0, "{:.2f}".format(output[0][0]))
        self.negatifArea.configure(state=tkinter.DISABLED)
        self.positifArea.configure(state=tkinter.NORMAL)
        self.positifArea.delete(0, tkinter.END)
        self.positifArea.insert(0, "{:.2f}".format(output[1][0]))
        self.positifArea.configure(state=tkinter.DISABLED)
        if np.argmax(output)==1:
            self.sentimenArea.configure(state=tkinter.NORMAL)
            self.sentimenArea.delete(0, tkinter.END)
            self.sentimenArea.insert(0, "Positif")
            self.sentimenArea.configure(state=tkinter.DISABLED)
            self.deskripsi_sentimen = customtkinter.CTkLabel(master=self.frame_right,
                                                   text="Teks ulasan menunjukkan kepuasan terhadap pelayanan atau barang",
                                                    text_font = ("Calibri", 14),
                                                   height=10,
                                                   fg_color=("white", "gray38"),
                                                   bg_color = None,  # <- custom tuple-color
                                                   justify=tkinter.CENTER)
        else:
            self.sentimenArea.configure(state=tkinter.NORMAL)
            self.sentimenArea.delete(0, tkinter.END)
            self.sentimenArea.insert(0, "Negatif")
            self.sentimenArea.configure(state=tkinter.DISABLED)
            self.deskripsi_sentimen = customtkinter.CTkLabel(master=self.frame_right,
                                                   text="Teks ulasan menunjukkan kekecewaan terhadap pelayanan atau barang",
                                                    text_font = ("Calibri", 14),
                                                   height=10,
                                                   fg_color=("white", "gray38"),
                                                   bg_color = None,  # <- custom tuple-color
                                                   justify=tkinter.CENTER)
        self.deskripsi_sentimen.grid(column=0, row=5,columnspan=2, padx=10, pady=10)
        return

    def aplikasi_klasifikasi(self):
        for widget in self.frame_info.winfo_children():
            widget.grid_forget()
        for widget in self.frame_right.winfo_children():
            widget.grid_forget()
        self.frame_info.grid(row=1, column=0, columnspan=3, rowspan=4, pady=20, padx=20, sticky="nsew")
        #Content
        self.judul_aplikasi = customtkinter.CTkLabel(master=self.frame_right,
                                                   text="APLIKASI KLASIFIKASI TEKS ULASAN",
                                                    text_font = ("Calibri", 18),
                                                   height=10,
                                                   fg_color=("white", "gray38"),
                                                   bg_color = None,  # <- custom tuple-color
                                                   justify=tkinter.CENTER)
        self.judul_aplikasi.grid(column=0, row=0,columnspan=2, padx=10, pady=10)
        
        self.pilihbobotoptimal_parameter=[]    
        for root, dirs, files in os.walk(PATH+'/Hasil'):
            for dir in dirs:
                for inti, folder, berkas in os.walk(PATH+'/Hasil/'+str(dir)):
                    for file_ in berkas:
                        x=file_.split('_')[:-1]
                        x.append('_'.join(file_.split('_')[-1:]).split('.')[0])
                        if x[0]=='bobot' and x[1]=='optimal':
                            self.pilihbobotoptimal_parameter.append(file_)

        self.tf_pilih_bobot=tkinter.StringVar()
        self.label_bobot_dipakai = customtkinter.CTkLabel(master=self.frame_info,
                                                   text="Pilih bobot optimal :",
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT) 
        self.label_bobot_dipakai.grid(row=0, column=0,pady=10, padx=20,sticky="w")   
        self.list_bobot_dipakai = tkinter.ttk.Combobox(self.frame_info,values=self.pilihbobotoptimal_parameter,state='readonly',textvariable=self.tf_pilih_bobot,width=35)
        self.list_bobot_dipakai.current(self.pilihbobotoptimal_parameter.index('bobot_optimal_256_0.01_25.txt'))
        self.list_bobot_dipakai.grid(row=0, column=1,pady=10, padx=20,sticky="w")

        self.textArea = customtkinter.CTkEntry(master=self.frame_info, text_font = ("Calibri", 12),
                                                   height=100,
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.textArea.grid(row=1, column=0, columnspan=2,pady=10, padx=20, sticky="nsew")

        self.positif_label = customtkinter.CTkLabel(master=self.frame_info,
                                                   text="Positif :",
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.positif_label.grid(row=3, column=0,pady=10, padx=20,sticky="w")

        self.positifArea = customtkinter.CTkEntry(master=self.frame_info,
                                            width=120,
                                            placeholder_text="")
        self.positifArea.grid(row=3, column=1, pady=10, padx=20, sticky="we")
        self.positifArea.configure(state="readonly")

        self.negatif_label = customtkinter.CTkLabel(master=self.frame_info,
                                                   text="Negatif :",
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.negatif_label.grid(row=4, column=0,pady=10, padx=20,sticky="w")

        self.negatifArea = customtkinter.CTkEntry(master=self.frame_info,
                                            width=120,
                                            placeholder_text="")
        self.negatifArea.grid(row=4, column=1, pady=10, padx=20, sticky="we")
        self.negatifArea.configure(state="readonly")

        self.sentimen_label = customtkinter.CTkLabel(master=self.frame_info,
                                                   text="Hasil Klasifikasi :",
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.sentimen_label.grid(row=5, column=0,pady=10, padx=20,sticky="w")

        self.sentimenArea = customtkinter.CTkEntry(master=self.frame_info,
                                            width=120,
                                            placeholder_text="")
        self.sentimenArea.grid(row=5, column=1, pady=10, padx=20, sticky="we")
        self.sentimenArea.configure(state="readonly")

        self.btn_prediksi = customtkinter.CTkButton(master=self.frame_info,height=25,
                                                       text="Klasifikasi",
                                                       command=threading.Thread(target=self.prediksi).start())
        self.btn_prediksi.grid(row=2, column=0,columnspan=2, pady=10, padx=20, sticky="we")
    
    def view_developer(self):
        for widget in self.frame_info.winfo_children():
            widget.grid_forget()
        for widget in self.frame_right.winfo_children():
            widget.grid_forget()
        self.frame_info.grid(row=1, column=0, columnspan=2, rowspan=4, pady=10, padx=20, sticky="nsew")

        #picture
        self.fiqih_image = ImageTk.PhotoImage(Image.open(PATH + "/image/081711233003.png").resize((135, 180), Image.ANTIALIAS))

        #Content
        self.judul_developer = customtkinter.CTkLabel(master=self.frame_right,
                                                   text="DEVELOPER",
                                                    text_font = ("Calibri", 18),
                                                   height=10,
                                                   fg_color=("white", "gray38"),
                                                   bg_color = None,  # <- custom tuple-color
                                                   justify=tkinter.CENTER)
        self.judul_developer.grid(column=0, row=0,columnspan=2, padx=10, pady=10)
        self.label_info_1 = customtkinter.CTkLabel(master=self.frame_info,
                                                   text="Judul : Klasifikasi Teks Ulasan dengan Hybrid Convolutional Neural Network \n       dan Gated Recurrent Unit \n"+
                                                        "Nama : Fiqih Fathor Rachim \n" +
                                                        "NIM : 081711233003 \n" +
                                                        "Pembimbing I : Auli Damayanti, S.Si, M.Si \n"+
                                                        "Pembimbing II : Drs. Edi Winarko, M.Cs \n" +
                                                        "Email : fiqih.fathor.rachim-2017@fst.unair.ac.id",
                                                    text_font = ("Calibri", 12),
                                                   height=100,
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.fiqih_button = customtkinter.CTkButton(master=self.frame_info, image=self.fiqih_image, text="", width=135, height=180,
                                   corner_radius=10, fg_color="gray40", hover_color="gray25")
        
        #Configure Grid
        self.label_info_1.grid(column=0, row=0, sticky="nwe", padx=15, pady=15)
        self.fiqih_button.grid(row=1, column=0,padx=15, pady=15)

    def view_help(self):
        for widget in self.frame_info.winfo_children():
            widget.grid_forget()
        for widget in self.frame_right.winfo_children():
            widget.grid_forget()
        self.frame_info.grid(row=1, column=0, columnspan=2, rowspan=4, pady=10, padx=20, sticky="nsew")

        #Content
        self.judul_bantuan = customtkinter.CTkLabel(master=self.frame_right,
                                                   text="BANTUAN",
                                                    text_font = ("Calibri", 18),
                                                   height=10,
                                                   fg_color=("white", "gray38"),
                                                   bg_color = None,  # <- custom tuple-color
                                                   justify=tkinter.CENTER)
        self.judul_bantuan.grid(column=0, row=0,columnspan=2, padx=10, pady=10)
        with open(PATH+"/bantuan.txt",'r') as file_:
            teks=file_.read()
        file_.close
        self.bantuan=tkinter.scrolledtext.ScrolledText(self.frame_info, 
                                      wrap = tkinter.WORD, 
                                      width = 50, 
                                      height = 30, 
                                      font = ("Arial",
                                              9))
        self.bantuan.insert(tkinter.END,teks)
        self.bantuan.configure(state="disabled")
        #Configure Grid
        self.bantuan.grid(column=0, row=0, sticky="nwe", padx=15, pady=15)

    def isfloat(self,value):
        try:
            float(value)
            return True
        except:
            return False
    def validate_integer(self,key,why,cek='0'):
        if cek=='0':
            if why=="1" and not key.isdigit():
                return False
            else:
                return True
        elif cek=='1':
            if why=="1" and not self.isfloat(key):
                return False
            else:
                return True

    def view_ekstraksi(self):
        for widget in self.frame_info.winfo_children():
            widget.grid_forget()
        for widget in self.frame_right.winfo_children():
            widget.grid_forget()
        self.frame_info.grid(row=1, column=0, columnspan=2, rowspan=4, pady=20, padx=20, sticky="nsew")
        
        #Content
        self.judul_ekstraksi = customtkinter.CTkLabel(master=self.frame_right,
                                                   text="EKSTRAKSI INFORMASI TERSEMBUNYI TEKS ULASAN DENGAN CNN",
                                                    text_font = ("Calibri", 18),
                                                   height=10,
                                                   fg_color=("white", "gray38"),
                                                   bg_color = None,  # <- custom tuple-color
                                                   justify=tkinter.CENTER)
        self.judul_ekstraksi.grid(column=0, row=0,columnspan=3, padx=10, pady=10)
        self.label_info_1 = customtkinter.CTkLabel(master=self.frame_info,
                                                   text="" ,
                                                   height=100,
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.label_info_1.grid(column=0, row=0,columnspan=2, sticky="nwe", padx=15, pady=15)
        self.label_info_2 = customtkinter.CTkLabel(master=self.frame_info,
                                                   text="Progres data latih" ,
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.label_info_2.grid(column=0, row=1, sticky="nwe", padx=15, pady=15)
        self.label_info_3 = customtkinter.CTkLabel(master=self.frame_info,
                                                   text="Progres data uji" ,
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.label_info_3.grid(column=0, row=2, sticky="nwe", padx=15, pady=15)

        self.progressbar1 = customtkinter.CTkProgressBar(master=self.frame_info)
        self.progressbar1.grid(row=1, column=1, sticky="ew", padx=15, pady=15)
        self.progressbar1.set(0)
        self.progressbar2 = customtkinter.CTkProgressBar(master=self.frame_info)
        self.progressbar2.grid(row=2, column=1, sticky="ew", padx=15, pady=15)
        self.progressbar2.set(0)

        # ============ frame_right ============

        self.lihat_data_latih = customtkinter.CTkButton(master=self.frame_right,
                                                       height=25,
                                                       text="Lihat Data Latih",
                                                       command=lambda : self.lihat_data(data="latih"))
        self.lihat_data_latih.grid(row=1, column=2, pady=10, padx=20, sticky="we")
        self.lihat_data_uji = customtkinter.CTkButton(master=self.frame_right,
                                                       height=25,
                                                       text="Lihat Data Uji",
                                                       command=lambda : self.lihat_data(data="uji"))
        self.lihat_data_uji.grid(row=2, column=2, pady=10, padx=20, sticky="we")

        self.label_kernel = customtkinter.CTkLabel(master=self.frame_right,
                                                   text="Banyak kernel [1-1000] :",
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.label_kernel.grid(row=5, column=0,pady=10, padx=20,sticky="w")

        self.entry_kernel = customtkinter.CTkEntry(master=self.frame_right,
                                            width=120)
        cek_integer=self.register(self.validate_integer)
        self.entry_kernel.config(validate="key",validatecommand=(cek_integer,'%P','%d',))
        self.entry_kernel.grid(row=6, column=0, columnspan=2, pady=10, padx=20, sticky="we")

        self.label_baris_kernel = customtkinter.CTkLabel(master=self.frame_right,
                                                   text="Ukuran baris kernel [3-10] :",
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.label_baris_kernel.grid(row=7, column=0,pady=10, padx=20,sticky="w")

        self.entry_baris_kernel = customtkinter.CTkEntry(master=self.frame_right,
                                            width=120)
        self.entry_baris_kernel.config(validate="key",validatecommand=(cek_integer,'%P','%d'))
        self.entry_baris_kernel.grid(row=8, column=0, columnspan=2, pady=10, padx=20, sticky="we")

        self.lihat_bobot_cnn = customtkinter.CTkButton(master=self.frame_right,
                                                       height=25,
                                                       text="Lihat Bobot CNN",
                                                       command=self.view_bobot_cnn)
        self.lihat_bobot_cnn.grid(row=9, column=0, pady=10, padx=20, sticky="we")
        self.lihat_input_matriks = customtkinter.CTkButton(master=self.frame_right,
                                                       height=25,
                                                       text="Lihat Input Matriks",
                                                       command=self.view_input_matriks)
        self.lihat_input_matriks.grid(row=9, column=1, pady=10, padx=20, sticky="we")

        list_kernel=""
        for root, dirs, files in os.walk(PATH+'/Hasil/'):
            for dir in dirs:
                list_kernel = list_kernel+dir+"\n"
        self.label_kernel_dilatih = customtkinter.CTkLabel(master=self.frame_right,
                                                   text="Kernel yang telah dilatih :",
                                                   fg_color=("white", "gray38"),
                                                   justify=tkinter.LEFT)
        self.label_kernel_dilatih.grid(row=3, column=2,pady=10, padx=20,sticky="w")

        self.label_info_dilatih = customtkinter.CTkLabel(master=self.frame_right,
                                                   height=200,
                                                   fg_color=("white", "gray38"),
                                                   text=list_kernel,
                                                   justify=tkinter.LEFT)
        self.label_info_dilatih.grid(column=2, row=4,rowspan=5, sticky="nwe", padx=15, pady=15)
        self.proses = customtkinter.CTkButton(master=self.frame_right,
                                                text="Proses",
                                                command= threading.Thread(target=self.cnn_maju).start)
        self.proses.grid(row=9, column=2,rowspan=1, columnspan=1, pady=20, padx=20, sticky="we")

    def handler_range(self,event):
        temp = self.tf_kernel.get()
        self.label_hidden_node.configure(text="Masukkan jumlah hidden node [1-"+ str(temp) +"] : ")
        
    def view_gru(self):
        for widget in self.frame_info.winfo_children():
            widget.grid_forget()
        for widget in self.frame_right.winfo_children():
            widget.grid_forget()
        self.frame_info.grid(row=1, column=0, columnspan=2, rowspan=4, pady=10, padx=20, sticky="nsew")
        
        #Content
        self.judul_gru = customtkinter.CTkLabel(master=self.frame_right,
                                                   text="PELATIHAN TEKS ULASAN DENGAN GRU",
                                                    text_font = ("Calibri", 18),
                                                   height=10,
                                                   fg_color=("white", "gray38"),
                                                   bg_color = None,  # <- custom tuple-color
                                                   justify=tkinter.CENTER)
        self.judul_gru.grid(column=0, row=0,columnspan=3, padx=10, pady=10)
        global label_info_1
        self.column_label=('iterasi','learning_rate','loss')
        label_info_1 = tkinter.ttk.Treeview(master=self.frame_info,
                                                   columns=self.column_label ,
                                                   show='headings')
        label_info_1.heading('iterasi', text='Iterasi')
        label_info_1.heading('learning_rate', text='Learning rate')
        label_info_1.heading('loss', text='Loss')
        label_info_1.grid(column=0, row=0,rowspan=1,columnspan=2, sticky="nwe", padx=15, pady=10)
        self.scrollbar = tkinter.ttk.Scrollbar(self.frame_info, orient=tkinter.VERTICAL, command=label_info_1.yview)
        label_info_1.configure(yscroll=self.scrollbar.set)
        self.scrollbar.grid(row=0, column=3, sticky='ns',pady=10)

        self.label_info_uji = customtkinter.CTkLabel(master=self.frame_info,
                                                   text="Akurasi uji validasi" ,
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.label_info_uji.grid(column=0, row=1, sticky="nwe", padx=15, pady=10)
        self.entry_uji_val = customtkinter.CTkEntry(master=self.frame_info,
                                            placeholder_text="")
        self.entry_uji_val.grid(row=1, column=1, pady=10, padx=20, sticky="we")
        self.entry_uji_val.configure(state=tkinter.DISABLED)

        # ============ frame_right ============

        self.lihat_bobot_optimal = customtkinter.CTkButton(master=self.frame_right,
                                                       height=25,
                                                       text="Lihat bobot optimal",
                                                       command=self.lihat_bobot_optimal_func)
        self.lihat_bobot_optimal.grid(row=2, column=2, pady=10, padx=20, sticky="we")
        self.testing_data_uji = customtkinter.CTkButton(master=self.frame_right,
                                                       height=25,
                                                       text="Test data uji",
                                                       command=self.view_testing_data)
        self.testing_data_uji.grid(row=3, column=2, pady=10, padx=20, sticky="we")

        self.label_kernel = customtkinter.CTkLabel(master=self.frame_right,
                                                   text="Pilih input matriks hasil CNN :",
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.label_kernel.grid(row=6, column=0,pady=10, padx=20,sticky="w")
        OptionKernel=[]
        for root, dirs, files in os.walk(PATH+'/Hasil/'):
            for dir in dirs:
                OptionKernel.append(int(dir))
        
        self.tf_kernel=tkinter.IntVar()
        self.tf_hidden=tkinter.StringVar()
        self.tf_lr=tkinter.StringVar()
        self.tf_iterasi=tkinter.StringVar()
               
        self.list_kernel = tkinter.ttk.Combobox(self.frame_right,values=OptionKernel,state='readonly',textvariable=self.tf_kernel)
        self.list_kernel.current(0)
        self.list_kernel.grid(row=6, column=1,pady=10, padx=20,sticky="w")
        self.list_kernel.bind('<<ComboboxSelected>>', self.handler_range)
        cek_integer=self.register(self.validate_integer)
        self.label_hidden_node = customtkinter.CTkLabel(master=self.frame_right,
                                                   text="Masukkan jumlah hidden node [1-"+str(self.tf_kernel.get())+" ]:",
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT,anchor="w")
        self.label_hidden_node.grid(row=7, column=0,pady=10, padx=20,sticky="w")

        self.entry_hidden_node = customtkinter.CTkEntry(master=self.frame_right,
                                            width=120,textvariable=self.tf_hidden)
        self.entry_hidden_node.grid(row=7, column=1, columnspan=1, pady=10, padx=20, sticky="we")
        self.entry_hidden_node.config(validate="key",validatecommand=(cek_integer,'%P','%d'))

        self.label_lr = customtkinter.CTkLabel(master=self.frame_right,
                                                   text="Masukkan nilai learning rate [0-1] : ",
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.label_lr.grid(row=8, column=0,pady=10, padx=20,sticky="w")

        self.entry_lr = customtkinter.CTkEntry(master=self.frame_right,
                                            width=120,textvariable=self.tf_lr)
        self.entry_lr.grid(row=8, column=1, columnspan=1, pady=10, padx=20, sticky="we")
        self.entry_lr.config(validate="key",validatecommand=(cek_integer,'%P','%d','1'))

        self.label_iterasi = customtkinter.CTkLabel(master=self.frame_right,
                                                   text="Masukkan jumlah iterasi [1-1000] : ",
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.label_iterasi.grid(row=9, column=0,pady=10, padx=20,sticky="w")

        self.entry_iterasi = customtkinter.CTkEntry(master=self.frame_right,
                                            width=120,textvariable=self.tf_iterasi)
        self.entry_iterasi.grid(row=9, column=1, columnspan=1, pady=10, padx=20, sticky="we")
        self.entry_iterasi.config(validate="key",validatecommand=(cek_integer,'%P','%d'))

        self.proses_gru = customtkinter.CTkButton(master=self.frame_right,
                                                text="Proses",
                                                command=threading.Thread(target=self.proses_gru).start)
        self.proses_gru.grid(row=10, column=2,rowspan=1, columnspan=1, pady=10, padx=20, sticky="we")
    
    def view_testing_data(self):
        self.window_testing_data = customtkinter.CTkToplevel(self)
        self.window_testing_data.geometry("500x300")
        self.window_testing_data.title("Pilih Parameter Test Data Uji")
        self.frame_testing = customtkinter.CTkFrame(master=self.window_testing_data,
                               width=600,
                               height=400,
                               corner_radius=10)
        self.frame_testing.grid(row=0, column=0, sticky="nsew",padx=10,pady=10)

        self.label_pilih_kernel_test = customtkinter.CTkLabel(master=self.frame_testing,
                                                   text="Pilih kernel pada proses CNN :",
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.label_pilih_kernel_test.grid(row=0, column=0,pady=10, padx=20,sticky="w")
        OptionKernel_test=[]
        for root, dirs, files in os.walk(PATH+'/Hasil/'):
            for dir in dirs:
                OptionKernel_test.append(int(dir))
        
        self.temp_a_test=tkinter.StringVar()
        self.temp_b_test=tkinter.StringVar()
        self.temp_c_test=tkinter.StringVar()
        self.temp_d_test=tkinter.StringVar()

        self.list_kernel = tkinter.ttk.Combobox(self.frame_testing,values=OptionKernel_test,state='readonly',textvariable=self.temp_a_test)
        self.list_kernel.current(0)
        self.list_kernel.grid(row=0, column=1,pady=10, padx=20,sticky="w")
        self.list_kernel.bind('<<ComboboxSelected>>', self.view_hidden_node)

        self.label_hidden_node = customtkinter.CTkLabel(master=self.frame_testing,
                                                   text="Pilih jumlah hidden node pada proses GRU :",
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.label_hidden_node.grid(row=1, column=0,pady=10, padx=20,sticky="w")
        self.list_hidden_node = tkinter.ttk.Combobox(self.frame_testing,state='readonly',textvariable=self.temp_b_test)
        self.list_hidden_node.grid(row=1, column=1,pady=10, padx=20,sticky="w")
        self.list_hidden_node.bind('<<ComboboxSelected>>', self.view_lr)

        self.label_lr = customtkinter.CTkLabel(master=self.frame_testing,
                                                   text="Pilih learning rate :",
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.label_lr.grid(row=2, column=0,pady=10, padx=20,sticky="w")
        self.list_lr = tkinter.ttk.Combobox(self.frame_testing,state='readonly',textvariable=self.temp_c_test)
        self.list_lr.grid(row=2, column=1,pady=10, padx=20,sticky="w")
        self.list_lr.bind('<<ComboboxSelected>>', self.view_iterasi)

        self.label_iterasi = customtkinter.CTkLabel(master=self.frame_testing,
                                                   text="Pilih iterasi :",
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.label_iterasi.grid(row=3, column=0,pady=10, padx=20,sticky="w")
        self.list_iterasi = tkinter.ttk.Combobox(self.frame_testing,state='readonly',textvariable=self.temp_d_test)
        self.list_iterasi.grid(row=3, column=1,pady=10, padx=20,sticky="w")

        self.label_info_testing = customtkinter.CTkLabel(master=self.frame_testing,
                                                   text="Akurasi testing" ,
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.label_info_testing.grid(column=0, row=4, sticky="nwe", padx=15, pady=15)
        self.entry_uji_testing = customtkinter.CTkEntry(master=self.frame_testing,
                                            placeholder_text="")
        self.entry_uji_testing.grid(row=4, column=1, pady=10, padx=20, sticky="we")
        self.entry_uji_testing.configure(state=tkinter.DISABLED)

        self.proses_testing_data= customtkinter.CTkButton(master=self.frame_testing,
                                                text="Proses",
                                                command=self.proses_testing_func)
        self.proses_testing_data.grid(row=5, column=1,pady=10, padx=20,sticky="w")
    
    def proses_testing_func(self):
        a=self.temp_a_test.get()
        b=self.temp_b_test.get()
        c=self.temp_c_test.get()
        d=self.temp_d_test.get()
        if b=="" or c=="" or d=="":
            tkinter.messagebox.showwarning("Peringatan", "Silahkan pilih parameter dengan benar!")
            return
        with open(PATH +"/hasil/"+str(a)+"/bobot_optimal_"+str(b)+"_"+str(c)+"_"+str(d)+".txt","rb") as f:
            bobot_optimal=pickle.load(f)
        f.close()

        data_uji=[]
        target_uji=[]
        with open(PATH+"/hasil/"+str(a)+'/Vektor_input_data_uji_'+str(a)+'.txt','r') as file_:
             for data in file_:
                wadah=data[:-1]
                remove_char="[] "
                for char in remove_char:
                    wadah=wadah.replace(char,"")
                a_temp=wadah.split(',')
                b_temp=[]
                for char in a_temp:
                    b_temp.append([float(char)])
                data_uji.append(b_temp)
        file_.close()
        with open(PATH+"/data/target_test.txt",'r') as file_:
            for data in file_:
                wadah=data[:-1]
                target_uji.append(float(wadah))
        file_.close()

        #Pengujian data
        with open(PATH+"/hasil/"+str(a)+'/maxmindata.txt','rb') as file_:
            load=pickle.load(file_)
            max_data=load[0]
            min_data=load[1]
        file_.close()
        data_uji=np.array(data_uji)
        data_uji=data_uji.astype('float64')
        X_uji=normalisasi(data_uji,load)

        number_output=2
        uji=GRU(len(X_uji[0]), int(b), number_output,bobot_optimal)
        TP=0 #truepositif
        TN=0 #truenegatif
        FP=0 #falsepositif
        FN=0 #falsenegatif
        for data,target in zip(X_uji,target_uji):
            output=uji.forward(data)
            if target==1:
                if np.argmax(output)==target:
                    TP=TP+1
                else:
                    FN=FN+1
            else:
                if np.argmax(output)==target:
                    TN=TN+1
                else:
                    FP=FP+1
        akurasi=(TP+TN)/(TP+TN+FP+FN)*100
        #recall=TP/(TP+FN)
        #precision=TP/(TP+FP)
        self.entry_uji_testing.configure(state=tkinter.NORMAL)
        self.entry_uji_testing.delete(0,tkinter.END)
        self.entry_uji_testing.insert(tkinter.END,str(akurasi)+'%')
        self.entry_uji_testing.configure(state=tkinter.DISABLED)
        #print("Recall = ",recall)
        #print("Precison = ",precision)

    def proses_gru(self):
        self.entry_uji_val.configure(state=tkinter.NORMAL)
        self.entry_uji_val.delete(0,END)
        self.entry_uji_val.configure(state=tkinter.DISABLED)
        try:
            jmlh_kernel=int(self.tf_kernel.get())
            jmlh_hidden=int(self.tf_hidden.get())
            nilai_lr=float(self.tf_lr.get())
            jmlh_iterasi=int(self.tf_iterasi.get())
        except:
            tkinter.messagebox.showwarning("Peringatan", "Input tidak boleh kosong!")
            return
        
        if jmlh_hidden<0 or jmlh_hidden>jmlh_kernel :
            tkinter.messagebox.showwarning("Peringatan", "Jumlah hidden tidak sesuai dengan interval yang diijinkan!")
            return
        if nilai_lr>1 or nilai_lr<0:
            tkinter.messagebox.showwarning("Peringatan", "Nilai learning rate tidak sesuai dengan interval yang diijinkan!")
            return
        if jmlh_iterasi>1000 or jmlh_iterasi<=0:
            tkinter.messagebox.showwarning("Peringatan", "Jumlah iterasi tidak sesuai dengan interval yang diijinkan!")
            return
        data_latih=[]
        target_latih=[]
        with open(PATH+"/hasil/"+str(jmlh_kernel)+'/Vektor_input_data_latih_'+str(jmlh_kernel)+'.txt','r') as f:
            for data in f:
                wadah=data[:-1]
                remove_char="[] "
                for char in remove_char:
                    wadah=wadah.replace(char,"")
                a=wadah.split(',')
                b=[]
                for char in a:
                    b.append([float(char)])
                data_latih.append(b)
        
        with open(PATH+"/data/"+'target_train.txt','r') as g:
            for data in g:
                wadah=data[:-1]
                wadah=float(wadah)
                if wadah == 1:
                    target_latih.append([[0],[1]])
                else :
                    target_latih.append([[1],[0]])
        
        number_input=len(data_latih[0])
        cek_file_bobot='bobot_optimal_'+str(jmlh_hidden)+'_'+str(nilai_lr)+'_'+str(jmlh_iterasi)+'.txt'
        if path.exists(os.path.join(os.getcwd(), PATH+"/hasil/"+str(jmlh_kernel), cek_file_bobot))==True:
            cek_pelatihan=tkinter.messagebox.askyesno("Pertanyaan", "Anda mungkin telah melakukan pelatihan dengan parameter ini, ingin mengulangi pelatihan kembali?")
            if cek_pelatihan=="no":
                tkinter.messagebox.showinfo("Informasi", "Anda membatalkan pelatihan")
                return
        data_latih=np.array(data_latih)
        data_latih=data_latih.astype('float64')
        temp=maxmin(data_latih,jmlh_kernel)
        X_latih=normalisasi(data_latih, temp)
        number_output=2
        gru=GRU(number_input, jmlh_hidden, number_output)
        gru.train(X_latih,target_latih,nilai_lr,jmlh_iterasi,0.00001)
        self.window_plot = customtkinter.CTkToplevel(self)
        self.window_plot.geometry("700x700")
        self.fig = plt.Figure(figsize = (7, 7),
                 dpi = 100)
        self.plot1 = self.fig.add_subplot(111)
        self.plot1.plot(iterasi,eror)
        self.plot1.set_title("Grafik error terhadap iterasi dengan hidden node = "+str(jmlh_hidden)+", learning rate = "+str(nilai_lr))
        self.plot1.set_xlabel("Iterasi")
        self.plot1.set_ylabel("error")
        self.canvas = FigureCanvasTkAgg(self.fig,
                               master = self.window_plot)  
        self.canvas.get_tk_widget().grid(row=0, column=0,padx=10,pady=10)
        bobot_optimal=[gru.w,gru.b]
        with open(PATH+"/hasil/"+str(jmlh_kernel)+'/bobot_optimal_'+str(jmlh_hidden)+'_'+str(nilai_lr)+'_'+str(jmlh_iterasi)+'.txt','wb') as file_:           
            pickle.dump(bobot_optimal,file_)
        file_.close()
        target_latih_akurasi=[]
        with open(PATH+'/data/target_train.txt','r') as file_:
            for data in file_:
                wadah=data[:-1]
                target_latih_akurasi.append(float(wadah))
        file_.close()
                
        uji=GRU(len(X_latih[0]), jmlh_hidden, number_output,bobot_optimal)
        TP=0 #truepositif
        TN=0 #truenegatif
        FP=0 #falsepositif
        FN=0 #falsenegatif
        for data,target in zip(X_latih,target_latih_akurasi):
            output=uji.forward(data)
            if target==1:
                if np.argmax(output)==target:
                    TP=TP+1
                else:
                    FN=FN+1
            else:
                if np.argmax(output)==target:
                    TN=TN+1
                else:
                    FP=FP+1
        akurasi=(TP+TN)/(TP+TN+FP+FN)*100
        #recall=TP/(TP+FN)
        #precision=TP/(TP+FP)
        self.entry_uji_val.configure(state=tkinter.NORMAL)
        self.entry_uji_val.insert(END,str(akurasi)+"%")
        self.entry_uji_val.configure(state=tkinter.DISABLED)
        #print("Recall = ",recall)
        #print("Precison = ",precision)
        return

    def lihat_bobot_optimal_func(self):
        self.window_bobot_optimal = customtkinter.CTkToplevel(self)
        self.window_bobot_optimal.title("Pilih Parameter Bobot Optimal")
        self.window_bobot_optimal.geometry("500x250")
        self.frame = customtkinter.CTkFrame(master=self.window_bobot_optimal,
                               width=600,
                               height=400,
                               corner_radius=10)
        self.frame.grid(row=0, column=0, sticky="nsew",padx=10,pady=10)

        self.label_kernel = customtkinter.CTkLabel(master=self.frame,
                                                   text="Pilih jumlah kernel pada proses CNN :",
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.label_kernel.grid(row=0, column=0,pady=10, padx=20,sticky="w")
        OptionKernel=[]
        for root, dirs, files in os.walk(PATH+'/Hasil/'):
            for dir in dirs:
                OptionKernel.append(int(dir))
        
        self.temp_a=tkinter.StringVar()
        self.temp_b=tkinter.StringVar()
        self.temp_c=tkinter.StringVar()
        self.temp_d=tkinter.StringVar()

        self.list_kernel = tkinter.ttk.Combobox(self.frame,values=OptionKernel,state='readonly',textvariable=self.temp_a)
        self.list_kernel.current(0)
        self.list_kernel.grid(row=0, column=1,pady=10, padx=20,sticky="w")
        self.list_kernel.bind('<<ComboboxSelected>>', self.view_hidden_node)

        self.label_hidden_node = customtkinter.CTkLabel(master=self.frame,
                                                   text="Pilih jumlah hidden node pada proses GRU :",
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.label_hidden_node.grid(row=1, column=0,pady=10, padx=20,sticky="w")
        self.list_hidden_node = tkinter.ttk.Combobox(self.frame,state='readonly',textvariable=self.temp_b)
        self.list_hidden_node.grid(row=1, column=1,pady=10, padx=20,sticky="w")
        self.list_hidden_node.bind('<<ComboboxSelected>>', self.view_lr)

        self.label_lr = customtkinter.CTkLabel(master=self.frame,
                                                   text="Pilih learning rate :",
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.label_lr.grid(row=2, column=0,pady=10, padx=20,sticky="w")
        self.list_lr = tkinter.ttk.Combobox(self.frame,state='readonly',textvariable=self.temp_c)
        self.list_lr.grid(row=2, column=1,pady=10, padx=20,sticky="w")
        self.list_lr.bind('<<ComboboxSelected>>', self.view_iterasi)

        self.label_iterasi = customtkinter.CTkLabel(master=self.frame,
                                                   text="Pilih iterasi :",
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.label_iterasi.grid(row=3, column=0,pady=10, padx=20,sticky="w")
        self.list_iterasi = tkinter.ttk.Combobox(self.frame,state='readonly',textvariable=self.temp_d)
        self.list_iterasi.grid(row=3, column=1,pady=10, padx=20,sticky="w")

        self.lihat_bobot_optimal= customtkinter.CTkButton(master=self.frame,
                                                text="Lihat bobot",
                                                command=self.nilai_bobot_optimal)
        self.lihat_bobot_optimal.grid(row=4, column=1,pady=10, padx=20,sticky="w")
    
    def nilai_bobot_optimal(self, e="wr",f="0"):
        a=self.temp_a.get()
        b=self.temp_b.get()
        c=self.temp_c.get()
        d=self.temp_d.get()
        if b=="" or c=="" or d=="" :
            tkinter.messagebox.showwarning("Peringatan","Silahkan pilih parameter dengan benar!")
            return
        if f=="1":
            self.window_nilai_bobot_optimal.destroy()    
        self.window_nilai_bobot_optimal = customtkinter.CTkToplevel(self)
        self.window_nilai_bobot_optimal.geometry("920x400")
        self.frame = customtkinter.CTkFrame(master=self.window_nilai_bobot_optimal,
                               width=600,
                               height=400,
                               corner_radius=10)
        self.frame.grid(row=0, column=0, sticky="nsew",padx=10,pady=10)
        with open(PATH +"/hasil/"+str(a)+"/"+"bobot_optimal_"+str(b)+"_"+str(c)+"_"+str(d)+".txt","rb") as f:
            bobot_optimal=pickle.load(f)
        f.close()
        #bobot
        self.label_bobot = customtkinter.CTkLabel(master=self.frame,
                                                   text="Bobot :",
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.label_bobot.grid(row=0, column=0,pady=10, padx=20,sticky="w")
        #wr
        self.bobot_wr= customtkinter.CTkButton(master=self.frame,
                                                text="Wr",
                                                command=lambda : self.nilai_bobot_optimal(e="wr",f="1"))
        self.bobot_wr.grid(row=1, column=0,pady=10, padx=20,sticky="w")
        
        #ur
        self.bobot_ur= customtkinter.CTkButton(master=self.frame,
                                                text="Ur",
                                                command=lambda : self.nilai_bobot_optimal(e="ur",f="1"))
        self.bobot_ur.grid(row=2, column=0,pady=10, padx=20,sticky="w")

        #wz
        self.bobot_wz= customtkinter.CTkButton(master=self.frame,
                                                text="Wz",
                                                command=lambda : self.nilai_bobot_optimal(e="wz",f="1"))
        self.bobot_wz.grid(row=3, column=0,pady=10, padx=20,sticky="w")

        #uz
        self.bobot_uz= customtkinter.CTkButton(master=self.frame,
                                                text="Uz",
                                                command=lambda : self.nilai_bobot_optimal(e="uz",f="1"))
        self.bobot_uz.grid(row=4, column=0,pady=10, padx=20,sticky="w")

        #wh
        self.bobot_wh= customtkinter.CTkButton(master=self.frame,
                                                text="Wh",
                                                command=lambda : self.nilai_bobot_optimal(e="w_h",f="1"))
        self.bobot_wh.grid(row=5, column=0,pady=10, padx=20,sticky="w")

        #uh
        self.bobot_uh= customtkinter.CTkButton(master=self.frame,
                                                text="Uh",
                                                command=lambda : self.nilai_bobot_optimal(e="u_h",f="1"))
        self.bobot_uh.grid(row=6, column=0,pady=10, padx=20,sticky="w")

        #wo
        self.bobot_wo= customtkinter.CTkButton(master=self.frame,
                                                text="Wo",
                                                command=lambda : self.nilai_bobot_optimal(e="wo",f="1"))
        self.bobot_wo.grid(row=7, column=0,pady=10, padx=20,sticky="w")

        #bias
        self.label_bias = customtkinter.CTkLabel(master=self.frame,
                                                   text="Bias :",
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)
        self.label_bias.grid(row=0, column=1,pady=10, padx=20,sticky="w")
        #r
        self.bias_r= customtkinter.CTkButton(master=self.frame,
                                                text="br",
                                                command=lambda : self.nilai_bobot_optimal(e="r",f="1"))
        self.bias_r.grid(row=1, column=1,pady=10, padx=20,sticky="w")

        #z
        self.bias_z= customtkinter.CTkButton(master=self.frame,
                                                text="bz",
                                                command=lambda : self.nilai_bobot_optimal(e="z",f="1"))
        self.bias_z.grid(row=2, column=1,pady=10, padx=20,sticky="w")

        #h
        self.bias_h= customtkinter.CTkButton(master=self.frame,
                                                text="bh",
                                                command=lambda : self.nilai_bobot_optimal(e="_h",f="1"))
        self.bias_h.grid(row=3, column=1,pady=10, padx=20,sticky="w")

        #o
        self.bias_o= customtkinter.CTkButton(master=self.frame,
                                                text="bo",
                                                command=lambda : self.nilai_bobot_optimal(e="o",f="1"))
        self.bias_o.grid(row=4, column=1,pady=10, padx=20,sticky="w")

        self.frame2 = customtkinter.CTkFrame(master=self.window_nilai_bobot_optimal,
                               width=600,
                               height=400,
                               corner_radius=10)
        self.frame2.grid(row=0, column=1, sticky="nsew",padx=10,pady=10)
        if e=="r" or e=="z" or e=="_h" or e=="o" :
            df = pd.DataFrame(bobot_optimal[1][e])
        else:
            df = pd.DataFrame(bobot_optimal[0][e])
        self.table = pt = Table(self.frame2, dataframe=df,
                        )
        pt.redraw()
        pt.show()
  
    def view_iterasi(self,event):
        self.list_iterasi.set("")
        self.list_iterasi["values"]=[]
        temp_lr=event.widget.get()
        for root, dirs, files in os.walk(PATH+'/Hasil/'+str(self.simpan_kernel)+"/"):
            option_iterasi=[]
            for file_ in files:
                x=file_.split('_')[:-1]
                x.append('_'.join(file_.split('_')[-1:]).split('.')[0])
                if x[0]=='bobot' and x[1]=='optimal' and x[2]==str(self.simpan_hidden) and x[3]==str(temp_lr):
                    if x[4] not in option_iterasi:
                        option_iterasi.append(x[4])
            self.list_iterasi["values"]=option_iterasi
        self.simpan_lr=temp_lr

    def view_lr(self,event):
        self.list_lr.set("")
        self.list_iterasi.set("")
        self.list_lr["values"]=[]
        self.list_iterasi["values"]=[]
        temp_hidden=event.widget.get()
        for root, dirs, files in os.walk(PATH+'/Hasil/'+str(self.simpan_kernel)+"/"):
            option_lr=[]
            for file_ in files:
                x=file_.split('_')[:-1]
                x.append('_'.join(file_.split('_')[-1:]).split('.')[0])
                if x[0]=='bobot' and x[1]=='optimal' and x[2]==str(temp_hidden):
                    if x[3] not in option_lr:
                        option_lr.append(x[3])
            self.list_lr["values"]=option_lr
        self.simpan_hidden=temp_hidden
    
    def view_hidden_node(self,event):
        self.list_lr.set("")
        self.list_hidden_node.set("")
        self.list_iterasi.set("")
        self.list_lr["values"]=[]
        self.list_hidden_node["values"]=[]
        self.list_iterasi["values"]=[]
        temp_kernel=event.widget.get()
        for root, dirs, files in os.walk(PATH+'/Hasil/'+str(temp_kernel)+"/"):
            option_hidden_node=[]
            for file_ in files:
                x=file_.split('_')[:-1]
                x.append('_'.join(file_.split('_')[-1:]).split('.')[0])
                if x[0]=='bobot' and x[1]=='optimal':
                    if x[2] not in option_hidden_node:
                        option_hidden_node.append(x[2])
            self.list_hidden_node["values"]=option_hidden_node
        self.simpan_kernel=temp_kernel
   
    def lihat_data(self, data):
        self.window = customtkinter.CTkToplevel(self)
        self.window.title("Data "+str(data))
        self.window.geometry("610x610")
        self.frame = customtkinter.CTkFrame(master=self.window,
                               width=400,
                               height=200,
                               corner_radius=10)
        self.frame.grid(row=0, column=0, sticky="nsew")
        # Add a Treeview widget

        if data == "latih":
            temp_data = pd.read_excel(PATH + "/data/Data pelatihan.xlsx",      # filepath here
                                                sheet_name = "Sheet1", # optional sheet name here
                                                engine = "openpyxl",
                                                header = None).values.tolist()
            self.sheet = Sheet(self.frame,
                           data =temp_data,width=600,height=600)
        elif data == "uji":
            temp_data = pd.read_excel(PATH + "/data/Data uji.xlsx",      # filepath here
                                                sheet_name = "Sheet1", # optional sheet name here
                                                engine = "openpyxl",
                                                header = None).values.tolist()
            self.sheet = Sheet(self.frame,data = temp_data,width=600,height=600)
        self.sheet.enable_bindings()
        self.sheet.readonly_rows(rows = [i for i in range(0,len(temp_data))], readonly = True, redraw = True)
        self.sheet.grid(row = 0, column = 0, sticky = "nswe")

    def view_bobot_cnn(self):
        filename = fd.askdirectory(
                    title='Pilih Folder Kernel',
                    initialdir=PATH+"/hasil/")
        x=path.basename(path.normpath(filename))
        if filename:
            try:
                with open(PATH+"/hasil/"+x+"/bobot_cnn_"+x+".txt","rb") as file:
                    bobot_optimal=pickle.load(file)
                file.close()
                self.window = customtkinter.CTkToplevel(self)
                self.window.title("Input Kernel")
                self.window.geometry("350x50")
                self.frame = customtkinter.CTkFrame(master=self.window,
                                width=200,
                                height=100,
                                corner_radius=10)
                self.frame.grid(row=0, column=0,columnspan=2, padx=10,sticky="nsew")
                self.btn_proses = customtkinter.CTkButton(master=self.frame,
                                                        height=25,
                                                        text="Lihat",
                                                        command=lambda :self.view_data_bobot(bobot_optimal,self.entry_kernel.get()))
                self.btn_proses.grid(row=0, column=1, pady=10, padx=20, sticky="we")
                
                self.entry_kernel = customtkinter.CTkEntry(master=self.frame,
                                                width=120,
                                                placeholder_text="Kernel ke-")
                self.entry_kernel.grid(row=0, column=0, pady=10, padx=20, sticky="we")
            except:
                tkinter.messagebox.showwarning("Peringatan", "Mohon untuk tidak mengubah jalur directory folder yang telah di sediakan!")
                return
    
    def view_data_bobot(self,bobot_optimal_cnn,idx_data):
        try :
            temp=int(idx_data)
            if int(idx_data)<0 or int(idx_data)>=len(bobot_optimal_cnn):
                tkinter.messagebox.showwarning("Peringatan", "Mohon input indeks mulai dari 0-"+str(len(bobot_optimal_cnn)-1)+"!")
            else:
                self.window_data = customtkinter.CTkToplevel(self)
                self.window_data.title("Kernel CNN ke-"+str(idx_data))
                self.window_data.geometry("610x310")
                self.frame_data = customtkinter.CTkFrame(master=self.window_data,
                                    width=600,
                                    height=600,
                                    corner_radius=10)
                self.frame_data.grid(row=0, column=0,columnspan=2, sticky="nsew")
                self.sheet = Sheet(self.frame_data,default_header="numbers",width=600,height=300)
                self.sheet.set_sheet_data(data=bobot_optimal_cnn[int(idx_data)].tolist())
                self.sheet.readonly_rows(rows = [i for i in range(0,len(bobot_optimal_cnn[int(idx_data)].tolist()))], readonly = True, redraw = True)
                self.sheet.enable_bindings()
                self.sheet.grid(row=0,column=0,columnspan=2, padx=10,pady=10)
        except:
            tkinter.messagebox.showwarning("Peringatan", "Inputan harus bilangan bulat!")
            return

    def view_input_matriks(self):
        filename = fd.askdirectory(
                    title='Pilih Folder Kernel untuk Melihat vektor input',
                    initialdir=PATH+"/hasil/")
        x=path.basename(path.normpath(filename))
        try:
            df_latih=[]
            with open(PATH+"/hasil/"+str(x)+'/Vektor_input_data_latih_'+str(x)+'.txt','r') as f:
                    for data in f:
                        wadah=data[:-1]
                        remove_char="[] "
                        for char in remove_char:
                            wadah=wadah.replace(char,"")
                        a=wadah.split(',')
                        b=[]
                        for char in a:
                            b.append([float(char)])
                        df_latih.append(b)
            f.close()
            df_uji=[]
            with open(PATH+"/hasil/"+str(x)+'/Vektor_input_data_uji_'+str(x)+'.txt','r') as f:
                    for data in f:
                        wadah=data[:-1]
                        remove_char="[] "
                        for char in remove_char:
                            wadah=wadah.replace(char,"")
                        a=wadah.split(',')
                        b=[]
                        for char in a:
                            b.append([float(char)])
                        df_uji.append(b)
            f.close()
            if filename :
                self.window_data1 = customtkinter.CTkToplevel(self)
                self.window_data1.title("Hasil Proses Ekstraksi CNN untuk Kernel Sebanyak "+str(x))
                self.window_data1.geometry("950x360")
                self.frame_data1 = customtkinter.CTkFrame(master=self.window_data1,
                                width=800,
                                height=600,
                                corner_radius=10)
                self.frame_data1.grid(row=0, column=0,columnspan=2, sticky="nsew")
                self.label_data_latih = customtkinter.CTkLabel(master=self.frame_data1,
                                                    text="Vektor input data latih :",
                                                    fg_color=("white", "gray38"),  # <- custom tuple-color
                                                    justify=tkinter.LEFT)
                self.label_data_latih.grid(row=0, column=0,pady=10, padx=20,sticky="w")
                self.label_data_latih = customtkinter.CTkLabel(master=self.frame_data1,
                                                    text="Vektor input data uji :",
                                                    fg_color=("white", "gray38"),  # <- custom tuple-color
                                                    justify=tkinter.LEFT)
                self.label_data_latih.grid(row=0, column=1,pady=10, padx=20,sticky="w")
                
                self.sheet_data_latih = Sheet(self.frame_data1,
                                                    default_header="numbers")
                self.sheet_data_latih.set_sheet_data(df_latih)
                self.sheet_data_latih.readonly_rows(rows = [i for i in range(0,len(df_latih))], readonly = True, redraw = True)
                self.sheet_data_latih.enable_bindings()
                self.sheet_data_latih.grid(row=1,column=0,columnspan=1,padx=20,pady=10)

                self.sheet_data_uji = Sheet(self.frame_data1,default_header="numbers")
                self.sheet_data_uji.set_sheet_data(df_uji)
                self.sheet_data_uji.readonly_rows(rows = [i for i in range(0,len(df_uji))], readonly = True, redraw = True)
                self.sheet_data_uji.enable_bindings()
                self.sheet_data_uji.grid(row=1,column=1,columnspan=1,padx=20,pady=10)
        except:
            tkinter.messagebox.showwarning("Peringatan", "Mohon untuk tidak mengubah jalur directory folder yang telah disediakan!")
            return

    def cnn_maju(self):
        try:
            banyak_kernel = int(self.entry_kernel.get())
            ukuran_baris_kernel=int(self.entry_baris_kernel.get())
        except:
            tkinter.messagebox.showwarning("Peringatan", "Input tidak boleh kosong!")
            return
                                                         
        if isinstance(banyak_kernel, int)==False:
            tkinter.messagebox.showwarning("Peringatan", "Banyak kernel harus bilangan bulat 1-1000!")
            return
        if banyak_kernel>1000 or banyak_kernel<1 :
            tkinter.messagebox.showwarning("Peringatan", "Banyak kernel harus bilangan bulat 1-1000!")
            return

        if isinstance(ukuran_baris_kernel, int)==False:
            tkinter.messagebox.showwarning("Peringatan", "Ukuran baris kernel harus bilangan bulat 3-10!")
            return 
        if ukuran_baris_kernel>10 or ukuran_baris_kernel<3 :
            tkinter.messagebox.showwarning("Peringatan", "Ukuran baris kernel harus bilangan bulat 3-10!")
            return       

        self.label_info_1.configure(text="1.Memuat data latih dan uji")
        if path.exists(PATH+'/data/train.txt')==True :
            f = open(PATH+'/data/train.txt', "r")
            data_latih=f.readlines()
            f.close()
        else:
            tkinter.messagebox.showwarning("Peringatan", "File teks ulasan 'train.txt' tidak ditemukan, pastikan file berada pada directory folder data")
            self.view_ekstraksi()
            return 
        #memuat dan membaca data uji
        if path.exists(PATH+'/data/train.txt')==True :
            g = open(PATH+'/data/train.txt', "r")
            data_uji=g.readlines()
            g.close()
        else:
            tkinter.messagebox.showwarning("Peringatan", "File teks ulasan 'test.txt' tidak ditemukan, pastikan file berada pada directory folder data")
            self.view_ekstraksi()
            return 0
        #inisialisasi parameter CNN
        self.label_info_1.configure(text="1.Memuat data latih dan uji \n"
                                        +"2. Menginisialisasi parameter \n"
                                    )
        z=38
        tkinter.messagebox.showinfo("Informasi", "Anda telah menginisialisasi kernel sebanyak "+str(banyak_kernel)+" dengan ukuran "+str(ukuran_baris_kernel)+"x"+str(z))

        #pembentukan input matriks dari teks ulasan
        self.label_info_1.configure(text="1.Memuat data latih dan uji \n"
                                        +"2. Menginisialisasi parameter \n"
                                        +"3. Pembentukan matriks representasi dengan One Hot Enconding \n"
                                    )
        matrix_latih=[]
        for row in data_latih:
            #pemanggilan fungsi dengan algoritma onehotencoding
            matrix_latih.append(onehotencoding(row, 500))
        matrix_uji=[]
        for row in data_uji:
            matrix_uji.append(onehotencoding(row, 500))

        if path.isdir(PATH+"/hasil/"+str(banyak_kernel))==True :
            file_check="bobot_cnn_"+str(banyak_kernel)+".txt"
            file_check1="Vektor_input_data_latih_"+str(banyak_kernel)+".txt"
            file_check2="Vektor_input_data_uji"+str(banyak_kernel)+".txt"
            if path.exists(os.path.join(os.getcwd(), PATH+"/hasil/"+str(banyak_kernel), file_check))==True:
                if path.exists(os.path.join(os.getcwd(), PATH+"/hasil/"+str(banyak_kernel), file_check1))==True:
                    if path.exists(os.path.join(os.getcwd(), PATH+"/hasil/"+str(banyak_kernel), file_check2))==True:
                        check_msg = messagebox.askyesno("Pertanyaan", "Anda mungkin telah melakukan ekstrasi dengan parameter ini, ingin mengulangi ekstraksi kembali?")
                        if check_msg=="yes":
                            self.label_info_1.configure(text="1.Memuat data latih dan uji \n"
                                        +"2. Menginisialisasi parameter \n"
                                        +"3. Pembentukan matriks representasi dengan One Hot Enconding \n"
                                        +"4. Ektraksi teks ulasan dengan CNN \n"
                                    )
                            bobot_cnn, vektor_input_data_latih,vektor_input_data_uji=self.umpan_maju(matrix_latih,matrix_uji,banyak_kernel,ukuran_baris_kernel,z)
                        elif check_msg=="no":
                            self.view_ekstraksi()
                            return
                    else:
                        self.label_info_1.configure(text="1.Memuat data latih dan uji \n"
                                        +"2. Menginisialisasi parameter \n"
                                        +"3. Pembentukan matriks representasi dengan One Hot Enconding \n"
                                        +"4. Ektraksi teks ulasan dengan CNN \n"
                                    ) 
                        bobot_cnn, vektor_input_data_latih,vektor_input_data_uji=self.umpan_maju(matrix_latih,matrix_uji,banyak_kernel,ukuran_baris_kernel,z)
                else:
                    self.label_info_1.configure(text="1.Memuat data latih dan uji \n"
                                        +"2. Menginisialisasi parameter \n"
                                        +"3. Pembentukan matriks representasi dengan One Hot Enconding \n"
                                        +"4. Ektraksi teks ulasan dengan CNN \n"
                                    ) 
                    bobot_cnn, vektor_input_data_latih,vektor_input_data_uji=self.umpan_maju(matrix_latih,matrix_uji,banyak_kernel,ukuran_baris_kernel,z)
            else:
                self.label_info_1.configure(text="1.Memuat data latih dan uji \n"
                                        +"2. Menginisialisasi parameter \n"
                                        +"3. Pembentukan matriks representasi dengan One Hot Enconding \n"
                                        +"4. Ektraksi teks ulasan dengan CNN \n"
                                    ) 
                bobot_cnn, vektor_input_data_latih,vektor_input_data_uji=self.umpan_maju(matrix_latih,matrix_uji,banyak_kernel,ukuran_baris_kernel,z)
        else:
            self.label_info_1.configure(text="1.Memuat data latih dan uji \n"
                                        +"2. Menginisialisasi parameter \n"
                                        +"3. Pembentukan matriks representasi dengan One Hot Enconding \n"
                                        +"4. Ektraksi teks ulasan dengan CNN \n"
                                    ) 
            bobot_cnn, vektor_input_data_latih,vektor_input_data_uji=self.umpan_maju(matrix_latih,matrix_uji,banyak_kernel,ukuran_baris_kernel,z)

        #Menyimpan hasil CNN
        os.mkdir(PATH+"/hasil/"+str(banyak_kernel))
        self.label_info_1.configure(text="1.Memuat data latih dan uji \n"
                                        +"2. Menginisialisasi parameter \n"
                                        +"3. Pembentukan matriks representasi dengan One Hot Enconding \n"
                                        +"4. Ektraksi teks ulasan dengan CNN \n"
                                        +"5. Menyimpan hasil proses CNN"
                                    )
        file1=open(PATH+"/hasil/"+str(banyak_kernel)+"/Vektor_input_data_latih_"+str(banyak_kernel)+".txt","w")
        for item in vektor_input_data_latih:
            file1.writelines("%s\n" % item)
        file1.close()

        file2=open(PATH+"/hasil/"+str(banyak_kernel)+"/Vektor_input_data_uji_"+str(banyak_kernel)+".txt","w")
        for item in vektor_input_data_uji:
            file2.writelines("%s\n" % item)
        file2.close()

        with open(PATH+"/hasil/"+str(banyak_kernel)+"/bobot_cnn_"+str(banyak_kernel)+".txt","wb") as file3:
            pickle.dump(bobot_cnn,file3)
        file3.close()
        tkinter.messagebox.showinfo("Informasi", "Ekstraksi telah selesai")
        return
        
    def umpan_maju(self,matrix_latih,matrix_uji,banyak_kernel,ukuran_baris,ukuran_kolom):
        """
        Tujuan : Umpan maju CNN
        Input : matrix_latih -> list matrix latih
                matrix_uji -> list matrix uji
                banyak_kernel
                ukuran_baris -> ukuran baris kernel
                ukuran_kolom -> ukuran kolom kernel
        output : output -> list hasil pooling
        """
        #inisialisasi CNN
        self.label_info_1.configure(text="1.Memuat data latih dan uji \n"
                                        +"2. Menginisialisasi parameter \n"
                                        +"3. Pembentukan matriks representasi dengan One Hot Enconding \n"
                                        +"4. Ektraksi teks ulasan dengan CNN \n"
                                        +"4.1 Ekstraksi pada data latih \n"
                                    )
        cnn=convolutional(banyak_kernel, ukuran_kolom, ukuran_kolom)
        vektor_input_data_latih=[]
        for i,row in enumerate(matrix_latih):
            #prosedur konvolusi
            hasil_konvolusi=cnn.konvolusi(row)
            #prosedur fungsi aktivasi ReLu
            hasil_relu=cnn.relu(hasil_konvolusi)
            #prosedur pooling
            hasil_pooling=cnn.pooling(hasil_relu)
            vektor_input_data_latih.append(hasil_pooling)
            self.progressbar1.set(i/len(matrix_latih))
        self.label_info_1.configure(text="1.Memuat data latih dan uji \n"
                                        +"2. Menginisialisasi parameter \n"
                                        +"3. Pembentukan matriks representasi dengan One Hot Enconding \n"
                                        +"4. Ektraksi teks ulasan dengan CNN \n"
                                        +"4.1 Ekstraksi pada data latih \n"
                                        +"4.2 Ekstraksi pada data uji \n"
                                    )
        vektor_input_data_uji=[]
        for i,row in enumerate(matrix_uji):
            hasil_konvolusi=cnn.konvolusi(row)
            hasil_relu=cnn.relu(hasil_konvolusi)
            hasil_pooling=cnn.pooling(hasil_relu)
            vektor_input_data_uji.append(hasil_pooling)
            self.progressbar2.set(i/len(matrix_uji))
        return cnn.filters,vektor_input_data_latih, vektor_input_data_uji

    def change_mode(self):
        if self.switch_2.get() == 1:
            customtkinter.set_appearance_mode("dark")
        else:
            customtkinter.set_appearance_mode("light")

    def on_closing(self, event=0):
        self.destroy()

    def start(self):
        self.mainloop()

if __name__ == "__main__":
    app = App()
    app.start()