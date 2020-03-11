import numpy as np
import os


class Network():
    def __init__(self,kol):
        self.kol = kol
        self.N = np.array([np.full((i, 2),0.0) for i in self.kol])
        self.W = np.array([np.random.rand(len(self.N[i]),len(self.N[i+1]))-0.5 for i in range(len(self.N)-1)])

    def trening(self,inputs,VD,k):
        self.N[0] = inputs
        self.VD = VD
        self.k = k

        for element in range(1,len(self.N)):
            self.N[element] = progon(self.N[element-1],self.N[element],self.W[element-1])
        self.N[len(self.N)-1] = OutFindError(self.VD,self.N[len(self.N)-1])

        for element in range(len(self.N),0):
            self.N[element-1] = FindError(self.N[element-1],self.N[element],self.W[element])

        for element in range(1,len(self.N)):
            self.W[element-1] = SaveError(self.N[element-1],self.N[element],self.W[element-1],self.k)
    def GetLeyer(self,stat_num):
        self.stat_num = stat_num
        return self.N[self.stat_num]
    def GetWeight(self,stat_num):
        self.stat_num = stat_num
        return self.W[self.stat_num]     
    def think(self,inputs):
        self.N[0] = inputs
        for element in range(1,len(self.N)):
            self.N[element] = progon(self.N[element-1],self.N[element],self.W[element-1])
        return self.N[len(self.N)-1]


    ###############SAVE WEIGHTS
    def SaveWeight(self,name):
        path = os.getcwd()
        try:
            os.mkdir(path+"/saves"+"/"+name)
        except(BaseException):
            if os.path.exists("saves") == False:
                os.mkdir(path+"/saves")
        for W_save in range(len(self.W)):
            np.save(path+"/saves/"+name+"/"+"Weight"+str(W_save),self.W[W_save])
    def LoadWeight(self,name):
        path = os.getcwd()
        try:
            for W_load in range(len(self.W)):
                self.W[W_load] = np.load(path+"/saves/"+name+"/"+"Weight"+str(W_load)+".npy")
        except(BaseException):
            return 0
    ###############SAVE WIGHTS

def act(x):
    return 1/(1 + np.exp(-x))

def progon(Li,Lo,W):
    VI = len(Li)
    VO = len(Lo)
    x = 0
    while x < VO:
        y = 0
        Lo[x][0] = 0
        while y < VI:
            Lo[x][0] += Li[y][0] * W[y][x]
            y += 1
        if Lo[x][0] > 1:
            Lo[x][0] = 1 + 0.1*(Lo[x][0] -1)
        if Lo[x][0] < 0:
            Lo[x][0] = 0.1 * Lo[x][0]
        x += 1
    return Lo

def OutFindError(IDL,N):
    V = len(IDL)
    x = 0
    while x < V:
        N[x][1] = IDL[x][0] - N[x][0] 
        if N[x][0] > 1 or N[x][0] < 0:
            N[x][1] = N[x][1] * 0.1
        x += 1
    return N

def FindError(Li,Lo,W):
    VO = len(Lo)
    VI = len(Li)
    x = 0
    while x < VI:
        Li[x][1] = 0
        y = 0
        while y < VO:
            Li[x][1] = Li[x][1] + W[x][y] * Lo[y][1]
            if Li[x][0] > 1 or Li[x][0] < 0:
                Li[x][1] = Li[x][1] * 0.1
            y += 1
        x += 1
    return Li

        

def SaveError(Li,Lo,W,k):
    VI = len(Li)
    VO = len(Lo)
    x = 0
    while x < VO:
        y = 0
        while y < VI:
            W[y][x] += k * Lo[x][1] * Li[y][0] 
            y += 1
        x += 1
    return W
    
