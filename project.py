import tkinter
from tkinter import *
from tkinter import filedialog
import pandas as pd
from tkinter import ttk
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

class AlgorithmsDeployment:

    def window1(self):
        window1 = Tk()
        label1 = Label(window1,text="Machine Learning algorithms deployment", bg="white",fg='black', font="ar 20 bold")
        label1.grid(row=2,column=2,padx=20,pady=20,ipadx=10,ipady=10)
        frame1=Frame(window1,width=100,highlightbackground='cyan',highlightthickness=5)
        frame1.grid(row=3,column=2,padx=200,pady=50,ipadx=20,ipady=20)
        def browsefiles():
            global filename,dataset
            filename = filedialog.askopenfilename(initialdir = "/",title = "select the file",filetypes = [("csv files","*.csv")])
            dataset = pd.read_csv(filename)
            Entry1.insert(0,filename)
        button1 = Button(frame1,text = " Browse ",bg='deeppink',fg="white",font="ar 13 bold",command = browsefiles)
        button1.grid(row=3,column=3,padx=20,pady=10)
        label2=Label(frame1,text="Upload Your Dataset",fg="black",font="ar 15 bold")
        label2.grid(row=0, column=2, padx=20, pady=10)
        Entry1=Entry(frame1,fg="black",font="ar 13 bold")
        Entry1.grid(row=3,column=2,padx=10, pady=10)
        button2 = Button(window1,text = "Next",background='deeppink',foreground="white",font="ar 13 bold",command = pages.window2)
        button2.grid(row=4,column=2,padx=10,pady=10)
        
        window1.configure(bg="white")
        window1.geometry("800x500")
        window1.title("Page1")
        window1.mainloop()
        
    def window2(self):
        window2 = Tk()

        label3 = Label(window2, text="Conform your dataset",bg="white", fg='black', font="ar 13 bold")
        label3.pack(pady=20)

        frame2 = Frame(window2,width=100,highlightbackground='cyan',highlightthickness=5)
        frame2.pack(padx=20)

        scroll1 = ttk.Scrollbar(frame2, orient='vertical')
        scroll1.pack(side=RIGHT, fill=Y)

        scroll2 = ttk.Scrollbar(frame2, orient='horizontal')
        scroll2.pack(side=BOTTOM, fill=X)

        global Treeviews
        Treeviews = ttk.Treeview(frame2, yscrollcommand=scroll1.set, xscrollcommand=scroll2.set)
        Treeviews.pack()

        scroll1.config(command=Treeviews.yview)
        scroll2.config(command=Treeviews.xview)

        Treeviews["column"] = list(dataset.columns)
        Treeviews["show"] = "headings"

        for column in Treeviews["column"]:
            Treeviews.heading(column, text=column)

        data_rows = dataset.to_numpy().tolist()
        for rows in data_rows:
            Treeviews.insert("", "end", values=rows)
        
        Treeviews.pack()

        button3 = Button(window2, text="NotConform",bg="gray",fg="white",font="ar 13 bold", command=pages.window1)
        button3.place(x=410,y=350,width=120)
        
        button4 = Button(window2,text = "Conform",background='deeppink',foreground="white",font="ar 13 bold",command = pages.window3)
        button4.place(x=710,y=350,width=140)

        window2.configure(bg="white")
        window2.title("Page2")
        window2.geometry("1200x500")
        window2.mainloop()
    
    def window3(self):
        window3 = Tk()
        
        label4 = Label(window3, text="Select the Features", fg='black',bg="white", font="ar 13 bold")
        label4.pack()
        
        label5 = Label(window3, text="Independent Variable", fg='black', bg="white" ,font="ar 13 bold")
        label5.pack(padx=30,side=LEFT,anchor="n")
        
        label5 = Label(window3, text="Dependent Variable", fg='#000000', bg="#ffffff", font=("Bookman Old Style", 15, 'bold'))
        label5.pack(padx=75,side=LEFT,anchor="n")
        
        frame3 = Frame(window3,width=10,highlightbackground='cyan',highlightthickness=5)
        frame3.place(x=65,y=75,width=220,height=150)
        
        scrollbar3 = Scrollbar(frame3)
        scrollbar3.pack(side=RIGHT,fill=BOTH )
        
        listbox1 = Listbox(frame3, selectmode=MULTIPLE)
        listbox1.config(yscrollcommand=scrollbar3.set)
        listbox1.place(width=205, height=150)

        scrollbar3.config(command=listbox1.yview)
        
        j=0
        for i in Treeviews["column"]:
            listbox1.insert(j, i)
            j = j + 1
        
        frame4 = Frame(window3,width=10,highlightbackground='cyan',highlightthickness=5)
        frame4.place(x=420, y=75, width=220, height=150)

        scrollbar4 = Scrollbar(frame4)
        scrollbar4.pack(side=RIGHT, fill=BOTH)

        listbox2 = Listbox(frame4, selectmode=SINGLE)
        listbox2.config(yscrollcommand=scrollbar4.set)
        listbox2.place(width=205, height=150)

        scrollbar4.config(command=listbox2.yview)

        j = 0
        for i in Treeviews["column"]:
            listbox2.insert(j, i)
            j = j + 1
        
        def independentVariables():
            global independent1, independent2
            independent1 = []
            independent2 = []
            label5 = Label(window3, text="Independent Variables :",background='white',foreground="black",font="ar 15 bold")
            label5.place(x=65, y=320)
            clicked = listbox1.curselection()
            z = 350
            for item in clicked:
                label6 = Label(window3, text=listbox1.get(item),bg="#ffffff")
                label6.place(x=65, y=z)
                independent1.append(item)
                independent2.append(listbox1.get(item))
                z = z + 15
                
        button5 = Button(window3,text="Select",background='deeppink',foreground="white",font="ar 15 bold",command=independentVariables)
        button5.place(x=140,y=255,width=100, height=30)
        
        def DependentVariables():
            global dependent1,dependent2
            dependent1 = []
            dependent2 = []
            label7 = Label(window3, text="Dependent Variables :",background='white',foreground="black",font="ar 15 bold")
            label7.place(x=420, y=320)
            click = listbox2.curselection()
            z = 350
            for items in click:
                label8 = Label(window3, text=listbox2.get(items),bg="white")
                label8.place(x=420, y=z)
                dependent1.append(items)
                dependent2.append(listbox2.get(items))
                z = z + 15


        button6 = Button(window3, text="Select",background='deeppink',foreground="white",font="ar 15 bold", command=DependentVariables)
        button6.place(x=495, y=255, width=100, height=30)

        button7 = Button(window3, text = "Conform",background='deeppink',foreground="white",font="ar 15 bold", command=pages.window4)
        button7.place(x=320, y=500, width=100, height=30)
        
        window3.configure(bg="white")
        window3.title("Page3")
        window3.geometry("800x1200")
        window3.mainloop()
        
    def window4(self):
        window4 = Tk()
        
        label8 = Label(window4, text="Select The Problem Type",fg='black', bg="white", font="ar 15 bold")
        label8.place(x=235,y=130,height=35)

        button8 = Button(window4, text="Regression", background='yellow',foreground="black",font="ar 15 bold",command=pages.regression)
        button8.place(x=90,y=250,width=130,height=40)

        button9 = Button(window4, text="Classification", background='deeppink',foreground="white",font="ar 15 bold",command=pages.classification)
        button9.place(x=290,y=250,width=140,height=40)

        button10 = Button(window4, text="Clustering", background='cyan',foreground="black",font="ar 15 bold", command=pages.Clustering)
        button10.place(x=490,y=250,width=130,height=40)
        
        window4.configure(bg="white")
        window4.title("Page4")
        window4.geometry("800x500")
        window4.mainloop()
        
    def regression(self):
        regressor = Tk()
        global algorithms
        algorithms="Regression"

        label19 = Label(regressor, text="Select Your Algorithms", fg='black', bg="cyan", font="ar 15 bold")
        label19.pack(pady=30)

        def comboclick(event):
            global Regression
            global accuracy
            
            x = dataset.iloc[:, independent1].values
            y = dataset.iloc[:, dependent1].values
            
            
            if (combobox1.get() == "Simple Linear Regression"):
                label20 = Label(regressor, text="You have selected Simple Linear Regression!",fg="green",font="ar 13 bold")
                label20.pack(pady=30)
                
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 3, random_state=0)

               
                Regression = LinearRegression()
                Regression.fit(x_train, y_train)
                accuracy = Regression.score(x_train, y_train)

            elif (combobox1.get() == "Multiple Linear Regression"):
                label21 = Label(regressor, text="You have selected Multiple Linear Regression!",fg="green",font="ar 13 bold")
                label21.pack(pady=30)
                global sc_X,sc_y
                sc_X = StandardScaler()
                sc_y = StandardScaler() 
                x = sc_X.fit_transform(x)
                y = sc_y.fit_transform(y)

                
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

                Regression = LinearRegression()
                Regression.fit(x_train, y_train)

                accuracy = Regression.score(x_train, y_train)

            elif (combobox1.get() == "Polynomial Regression"):
                label22 = Label(regressor, text="You have selected Polynomial Regression!",fg="green",font="ar 13 bold")
                label22.pack(pady=30)

                global polynomial_reg
                from sklearn.preprocessing import PolynomialFeatures
                polynomial_reg = PolynomialFeatures(degree = 2)
                x_ploy = polynomial_reg.fit_transform(x)
                Regression = LinearRegression()
                Regression.fit(x_ploy,y)


                
                accuracy = Regression.score(x_ploy,y)

            elif(combobox1.get()=="Supportive vector Regression"):
                label23 = Label(regressor, text="You have selected SVM Regression!",fg="green",font="ar 13 bold")
                label23.pack(pady=30)
                
                sc_X = StandardScaler()
                sc_y = StandardScaler() 
                x = sc_X.fit_transform(x)
                y = sc_y.fit_transform(y)

                Regression = SVR(kernel='rbf')
                Regression.fit(x,y)

                accuracy = Regression.score(x,y)

            elif(combobox1.get()=="Decision Tree Regression"):
                label24 = Label(regressor, text="You have selected Decision Tree Regression!",fg="green",font="ar 13 bold")
                label24.pack(pady=30)

                Regression = DecisionTreeRegressor(random_state=0)
                Regression.fit(x,y)

                accuracy = Regression.score(x,y)

            elif(combobox1.get()=="Random Forest Regression"):
                label25 = Label(regressor, text="You have selected Decision Tree Regression!",fg="green",font="ar 13 bold")
                label25.pack(pady=30)

                Regression = RandomForestRegressor(n_estimators=10,random_state=0)
                Regression.fit(x,y)

                accuracy=Regression.score(x,y)

        Algorithms = ["Simple Linear Regression", "Multiple Linear Regression", "Polynomial Regression","Supportive vector Regression","Decision Tree Regression","Random Forest Regression"]
    
        global combobox1
        combobox1 = ttk.Combobox(regressor, value=Algorithms)
        combobox1.config(width=50)
        combobox1.current(0)
        combobox1.bind("<<ComboboxSelected>>", comboclick)
        combobox1.pack(pady=30)

        button11 = Button(regressor, text='Train Model', bg="deeppink", fg="white",font="ar 13 bold", command=pages.window5)
        button11.pack(pady=20)

        regressor.configure(bg="white")
        regressor.title("Regression(page)")
        regressor.geometry("800x500")
        regressor.mainloop()
        
    def classification(self):
        classifier = Tk()

        global algorithms
        algorithms="Classification"
        global classification
        global acc

        label26 = Label(classifier, text="Select Your Algorithms", fg='black', bg="cyan", font="ar 15 bold")
        label26.pack(pady=30)
        

        def Combobox2(event):
            x = dataset.iloc[:, independent1].values
            y = dataset.iloc[:, dependent1].values

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

            global sc
            sc = StandardScaler()
            x_train = sc.fit_transform(x_train)
            x_test = sc.transform(x_test)
           
            global classification
            if(combobox2.get()=="Logistic Regression"):
                label27 = Label(classifier, text="You have Selected Logistic Regression",fg='green', font="ar 13 bold")
                label27.pack(pady=30)

                
                classification = LogisticRegression(random_state=0)
                classification.fit(x_train,y_train)

                y_pred = classification.predict(x_test)

                cm = confusion_matrix(y_test,y_pred)
                global acc
                acc = (sum(np.diag(cm))/len(y_test))

            elif(combobox2.get()=="Naive Bayes Classification"):
                label28 = Label(classifier, text="You have Selected Naive Bayes Algoritham",fg='green', font="ar 13 bold")
                label28.pack(pady=30)

                classification = GaussianNB()
                classification.fit(x_train ,y_train)

                y_pred = classification.predict(x_test)

                cm = confusion_matrix(y_test,y_pred)
                acc = (sum(np.diag(cm))/len(y_test))

            elif(combobox2.get()=="K-Nearest Neighbour Classification"):
                label29 = Label(classifier, text="You hve Selected K-Nearest Neighbour Algoritham",fg="green", font="ar 13 bold")
                label29.pack(pady=30)

                classification = KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2)
                classification.fit(x_train,y_train)

                y_pred = classification.predict(x_test)

                cm = confusion_matrix(y_test,y_pred)
                acc = sum(np.diag(cm))/len(y_test)

            elif(combobox2.get()=="SVM Classification"):
                label30 = Label(classifier, text="You have Selected SVM Classification",fg='green', font="ar 13 bold")
                label30.pack(pady=30)

                classification = SVC(kernel='rbf',random_state=0)
                classification.fit(x_train,y_train)

                y_pred=classification.predict(x_test)

                cm = confusion_matrix(y_test,y_pred)
                acc = sum(np.diag(cm))/len(y_test)

            elif(combobox2.get()=="Decision Tree Classification"):
                label31 = Label(classifier, text="You have Selected Decision Tree Classification",fg="green", font="ar 13 bold")
                label31.pack(pady=30)

                classification = DecisionTreeClassifier(criterion='entropy',random_state=0)
                classification.fit(x_train,y_train)

                y_pred = classification.predict(x_test)

                cm = confusion_matrix(y_test,y_pred)
                acc = sum(np.diag(cm))/len(y_test)

            elif(combobox2.get()=="Random Forest Classification"):
                label32 = Label(classifier, text="You have Selected Random Forest Classification",fg="green", font="ar 13 bold")
                label32.pack(pady=30)

                classification = RandomForestClassifier()
                classification.fit(x_train,y_train)

                y_pred = classification.predict(x_test)
                cm = confusion_matrix(y_test,y_pred)
                acc = sum(np.diag(cm))/len(y_test)

        Algorithms2 = ["Logistic Regression","Naive Bayes Classification","K-Nearest Neighbour Classification","SVM Classification","Decision Tree Classification","Random Forest Classification"]
        global combobox2
        combobox2 = ttk.Combobox(classifier, value=Algorithms2)
        combobox2.config(width=50)
        combobox2.current(0)
        combobox2.bind("<<ComboboxSelected>>", Combobox2)
        combobox2.pack(pady=30)

        button12 = Button(classifier, text='Train Model', bg="deeppink", fg="white",font="ar 13 bold",command=pages.window5)
        button12.pack(pady=30)

        classifier.configure(bg="white")
        classifier.geometry("800x500")
        classifier.title("Classification Algorithms")
        classifier.mainloop()
        
    def Clustering(self):
        cluster = Tk()
        global algorithms
        algorithms = "clustering"

        label33 = Label(cluster, text="Select Your Algorithms", fg='black', bg="cyan", font="ar 13 bold")
        label33.pack(pady=30)

        def Combobox3(event):
            x = dataset.iloc[:, independent1].values
            if(combobox3.get()=="KMeans Clustering"):

                wcss=[]

                for i in range(1,11):
                    kmeans = KMeans(n_clusters=i,random_state=0)
                    kmeans.fit(x)
                    wcss.append(kmeans.inertia_)
                def plot():
                    button92 = Button(cluster, text="Next",fg="black",bg="cyan",font="ar 13 bold", command=lambda:[kmeanspredict(),p.window7()])
                    button92.pack(padx=320, pady=90)

                    plt.figure(figsize = (8,5), dpi=50)
                    plt.scatter(range(1,11),wcss)
                    plt.plot(range(1,11),wcss)
                    plt.title("Elbow Method")
                    plt.xlabel("Number of Clusters")
                    plt.ylabel("WCSS")
                    plt.show()

                button13 = Button(cluster, text="View The Plot",fg="white",bg="deeppink",font="ar 13 bold", command=plot)
                button13.pack(padx=290, pady=30)

                label34 = Label(cluster,text="Enter the number of clusters :",fg='green',font="ar 13 bold")
                label34.place(x=215, y=285)

                entry31 = Entry(cluster)
                entry31.place(x=465,y=285,height=25,width=50)

                def kmeanspredict():
                    n = int(entry31.get())
                    kmeans = KMeans(n_clusters=n,random_state=0)
                    global y_value
                    y_value = kmeans.fit_predict(x)

            elif(combobox3.get()=="Hierarchical Clustering"):

                def plot1():
                    plt.figure(figsize=(8,5),dpi=50)
                    dendrogram = sch.dendrogram(sch.linkage(x,method='ward'))
                    plt.title("Dendrogram")
                    plt.xlabel("X-Values")
                    plt.ylabel("Euclidean Distance")
                    plt.show()

                button14 = Button(cluster, text="View The Plot",bg="deeppink",fg="white",font="ar 13 bold", command=plot1)
                button14.pack(padx=290, pady=30)

                label35 = Label(cluster, text="Enter the number of clusters :", fg="green",font="ar 13 bold")
                label35.place(x=215, y=285)

                entry31 = Entry(cluster)
                entry31.place(x=465, y=285, height=25, width=50)

                button15 = Button(cluster, text="Next",bg="cyan",fg="black",font="ar 13 bold", command=lambda: [hierarchy_predict(), pages.window7()])
                button15.pack(padx=320, pady=90,)

                def hierarchy_predict():
                    global y_value
                    n = int(entry31.get())
                    hc = AgglomerativeClustering(n_clusters=n,affinity="euclidean",linkage="ward")
                    y_value = hc.fit_predict(X)

        Algorithms3 = ["KMeans Clustering","Hierarchical Clustering"]
        global combobox3
        combobox3 = ttk.Combobox(cluster, value=Algorithms3)
        combobox3.config(width=50)
        combobox3.current(0)
        combobox3.bind("<<ComboboxSelected>>", Combobox3)
        combobox3.pack(pady=30)

        cluster.configure(bg="white")
        cluster.geometry("800x500")
        cluster.title("Clustering Window")
        cluster.mainloop()
        
    
        
    def window5(self):
        window5 = Tk()
        
        label35 = Label(window5, text="Features", fg='black', bg="cyan", font="ar 13 bold")
        label35.place(x=150, y=60, height=30)

        z=140
        global entry61, entries
        entries = []
        for i in independent2:
            label36 = Label(window5, text=i, fg='black', bg="white", font="ar 13 bold")
            label36.place(x=145, y=z)

            entry61 = Entry(window5)
            entry61.place(x=330, y=z, height=30, width=50)
            entries.append(entry61)
            z = z + 25

        label37 = Label(window5, text="Values", fg='black', bg="cyan", font="ar 13 bold")
        label37.place(x=330, y=60)

        def predict():
            global result
            result=[]
            for entry in entries:
                result.append(entry.get())

            if(algorithms=="Regression"):
                global pred
                if(combobox1.get()=="Polynomial Regression"):
                    pred = Regression.predict(polynomial_reg.transform([result]))
                    

                elif(combobox1.get()=="SVM Regression" or combobox1.get()=="Multiple Linear Regression" or combobox1.get()=="Decision Tree Regression" or combobox1.get()=="Random Forest Regression"):
                    pred = sc_y.inverse_transform(Regression.predict(sc_X.transform([result])))
                
    
            elif(algorithms=="Classification"):
                if (combobox2.get() == "Logistic Regression" or combobox2.get()=="K-Nearest Neighbour Classification" or combobox2.get()=="SVM Classification" or combobox2.get()=="Decision Tree Classification" or combobox2.get()=="Random Forest Classification" or combobox2.get()=="Naive Bayes Classification"):
                    pred = classification.predict(sc.transform([result]))
                   
        button16 = Button(window5, text="Predict Model", bg="deeppink", fg="white",font="ar 13 bold", command=lambda:[predict(),pages.window6()])
        button16.place(x=230,y=z+50)
        
        window5.configure(bg="white")
        window5.geometry("600x600")
        window5.title("Page5")
        window5.mainloop()
        
    def window6(self):
        window6 = Tk()
        
        if(algorithms=="Regression"):

            label38 = Label(window6,text="Summary", bg="cyan", fg="black", font="ar 15 bold")
            label38.place(x=330,y=100,width=100,height=30)

            label39 = Label(window6, text="Prediction Result", bg="white", fg="black",font="ar 15 bold")
            label39.place(x=140,y=200,width=200,height=30)

            label40 = Label(window6, text="Accuracy", bg="white", fg="black", font="ar 15 bold")
            label40.place(x=140, y=300, width=200, height=30)

            label41 = Label(window6, text=("{:.2f}".format(accuracy)), bg="white", fg="black",font="ar 15 bold")
            label41.place(x=460, y=300, width=100, height=30)

            pred1 = float(pred)
            pred1 = "{:.2f}".format(pred1)

            label42 = Label(window6, text=pred1, bg="black", fg="white",font="ar 15 bold")
            label42.place(x=460, y=200, width=100, height=30)

            button17 = Button(window6,text="Back",bg="yellow",fg="black",font="ar 15 bold",command=pages.window5)
            button17.place(x=70,y=400,width=90)

            button18 = Button(window6, text="Go To Home",bg="deeppink",fg="white",font="ar 15 bold",command=pages.window1)
            button18.place(x=530, y=400,width=140)

        elif(algorithms=="Classification"):
            label43 = Label(window6, text="Summary", bg="cyan", fg="black", font="ar 15 bold")
            label43.place(x=330, y=100, width=100, height=30)

            label44 = Label(window6, text="Prediction Result", bg="white", fg="black",font="ar 15 bold")
            label44.place(x=140, y=200, width=200, height=30)

            label45 = Label(window6, text="Accuracy", bg="white", fg="black" ,font="ar 15 bold")
            label45.place(x=140, y=300, width=200, height=30)

            label46 = Label(window6, text=("{:.2f}".format(acc)), bg="white", fg="black",font="ar 15 bold")
            label46.place(x=460, y=300, width=100, height=30)


            label47 = Label(window6, text=pred, bg="white", fg="black",font="ar 15 bold")
            label47.place(x=460, y=200, width=100, height=30)

            button19 = Button(window6, text="Back", bg="yellow", fg="black",font="ar 13 bold", command=pages.window5)
            button19.place(x=70, y=400, width=90)

            button20 = Button(window6, text="Go To Home", bg="deeppink", fg="white",font="ar 13 bold", command=pages.window1)
            button20.place(x=530, y=400, width=140)

        elif(algorithms=="clustering"):
            label48 = Label(window6, text="Predicted values from clustering", fg='#18068c', bg="#ffffff", font=("Bookman Old Style", 20, 'bold'))
            label48.pack(pady=20)

            frame = Frame(window6)
            frame.pack()

            scroll1 = ttk.Scrollbar(frame, orient='vertical')
            scroll1.pack(side=RIGHT, fill=Y)

            tview1 = ttk.Treeview(frame, yscrollcommand=scroll1.set)

            scroll1.config(command=tview1.yview)

            tview1["column"] = "Output"
            tview1["show"] = "headings"
            tview1.heading(tview1["column"],text=tview1["column"])

            out = y_value.tolist() 

            for value in out:
                tview1.insert("",END, values=value)

            tview1.pack()

            button21 = Button(window6, text="Back", bg="yellow", fg="black", font="ar 13 bold",command=pages.clustering)
            button21.place(x=70, y=400, width=90)

            button22 = Button(window6, text="Go To Home", bg="deeppink", fg="white",font="ar 13 bold", command=pages.window1)
            button22.place(x=530, y=400, width=140)
                         
        window6.configure(bg="white")
        window6.geometry("800x500")
        window6.title("Page6")
        window6.mainloop()

pages = AlgorithmsDeployment()
pages.window1()