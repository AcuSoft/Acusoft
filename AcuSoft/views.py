from django.http import HttpResponse
from django.template import Template, Context
from django.template import loader
from django.shortcuts import render

import os
import numpy as np
import csv
import pandas as pd
import math

clear = lambda: os.system("cls")
f = pd.read_csv("CoefAcu.csv")
pd.DataFrame(f)

reader = csv.reader(open('CoefAcu.csv'))

Materiales = {}
for row in reader:
   key = row[0]
   if key in Materiales:
       pass
   milistafloat = []
   for item in row[1:]:
       milistafloat.append(float(item))
   Materiales[key] = milistafloat

class room():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.v = x*y*z
        self.a = 2*(x*y+x*z+y*z)

    def Calcular(self,T, P, WF, WB, WL, WR, v, a, ssaW):
        self.sa    = T + P + WF + WB + WL + WR + ssaW
        self.RT60f = -0.161*v/(a*np.log(1-(self.sa/a)))
        self.RT60  = np.mean(self.RT60f)
        self.dev = (np.mean((self.RT60f-self.RT60)**2))**(1/2.0)

class wall():
    def __init__(self,Nombre,N,d1,d2):
        self.Nombre = Nombre
        self.Wall   = N
        self.Ancho  = d1
        self.Largo  = d2
        self.aW     = d1*d2

    def mat(self,m):
        self.MNombre  = m
        self.MCoefAcu = Materiales[self.MNombre] 

    def saWall(self):
        self.saW = self.aW*np.array(self.MCoefAcu)

    def restaWall(self,aOW):
        self.aOW = aOW
        self.aW -= self.aOW   

class objWall(wall):
    def mat(self,m):
        self.MNombre  = m
        self.MCoefAcu = Materiales[self.MNombre] 

    def saWall(self ):
        self.saW = self.aW*np.array(self.MCoefAcu)
    
    def ssaWall(self,ssaW):
        ssaW = ssaW

def home(request):
    Index_Loader = loader.get_template('index.html')
    Index_Template = Index_Loader.render({})
    return HttpResponse(Index_Template)

def RT60(request):
    clear()
    RT60_Loader = loader.get_template('RT60.html')
    RT60_Template = RT60_Loader.render({"Materiales":Materiales})
    return HttpResponse(RT60_Template)

import json
def RT60_resp(request):
    clear()
    if request.method == "GET":

        R = room (float(request.GET.get("x")),
                  float(request.GET.get("y")),
                  float(request.GET.get("z")))

        T  = wall("Techo",          "T",      R.x,R.z)
        P  = wall("Piso",           "P",      R.x,R.z)
        WF = wall("Pared Frontal",  "WF",     R.x,R.y)
        WB = wall("Pared trasera",  "WB",     R.x,R.y)
        WL = wall("Pared izquierda","WL",     R.z,R.y)
        WR = wall("Pared derecha",  "WR",     R.z,R.y)

        T.mat(request.GET.get("Material_T"))
        P.mat(request.GET.get("Material_P"))
        WF.mat(request.GET.get("Material_WF"))
        WB.mat(request.GET.get("Material_WB"))
        WL.mat(request.GET.get("Material_WL"))
        WR.mat(request.GET.get("Material_WR"))

        ssaW = np.zeros([1,6])
        objetos = {}
        sobjsaW = []

        if (request.GET.get('Extras') != None):       
            Extras = json.loads(request.GET.get('Extras'))

            n = 0
            i = 0
            while i < len(Extras):
                clear()
                obj = Extras[n]

                objx = objWall( obj[1],
                                obj[2], 
                                float(obj[3]), 
                                float(obj[4]))  
                    
                objx.mat(obj[5])
                    
                if   objx.Wall == "Techo":              T.restaWall(objx.aW)
                elif objx.Wall == "Piso":               P.restaWall(objx.aW)
                elif objx.Wall == "Pared Frontal":      WF.restaWall(objx.aW)
                elif objx.Wall == "Pared Trasera":      WB.restaWall(objx.aW)
                elif objx.Wall == "Pared Izquierda":    WL.restaWall(objx.aW)
                elif objx.Wall == "Pared Derecha":      WR.restaWall(objx.aW)            
                                            
                objx.saWall()
                
                objetos[n]=[objx.Nombre,objx.Wall,objx.aW,objx.MNombre,objx.saW]
                sobjsaW.append(np.around(objx.saW,2,None).tolist())
                                
                ssaW += np.array(objx.saW)
                n += 1
                i += 1
                                    
        else:
            Extras = None       

        T. saWall()
        P. saWall()
        WF.saWall()
        WB.saWall()
        WL.saWall()
        WR.saWall()

        saTnp = np.around(T .saW,2,None)
        saPnp = np.around(P .saW,2,None)
        saWFnp = np.around(WF .saW,2,None)
        saWBnp = np.around(WB .saW,2,None)
        saWLnp = np.around(WL .saW,2,None)
        saWRnp = np.around(WR .saW,2,None)

        saT = saTnp.tolist()
        saP = saPnp.tolist()
        saWF = saWFnp.tolist()
        saWB = saWBnp.tolist()
        saWL = saWLnp.tolist()
        saWR = saWRnp.tolist()

        R.Calcular( T.saW, 
                    P.saW, 
                    WF.saW, 
                    WB.saW, 
                    WL.saW, 
                    WR.saW, 

                    R.v, 
                    R.a,
                    ssaW )

        RT60f = np.around(R.RT60f,2,None)
        RT60f = RT60f.tolist()

    datos = {
            "x"   :R.x,
            "y"   :R.y,
            "z"   :R.z,
            
            "v"   :np.around(R.v,2,None),
            "a"   :np.around(R.a,2,None),

            "sT"  :np.around(T.aW,2,None),
            "sP"  :np.around(P.aW,2,None),
            "sWF" :np.around(WF.aW,2,None),
            "sWB" :np.around(WB.aW,2,None),
            "sWL" :np.around(WL.aW,2,None),
            "sWR" :np.around(WR.aW,2,None),

            "lT"  :T.Largo,
            "lP"  :P.Largo,
            "lWF" :WF.Largo,
            "lWB" :WB.Largo,
            "lWL" :WL.Largo,
            "lWR" :WR.Largo,

            "aT"  :T.Ancho,
            "aP"  :P.Ancho,
            "aWF" :WF.Ancho,
            "aWB" :WB.Ancho,
            "aWL" :WL.Ancho,
            "aWR" :WR.Ancho,
                        
            "matT"  :T.MNombre,
            "matP"  :P.MNombre,      
            "matWF" :WF.MNombre,
            "matWB" :WB.MNombre,
            "matWL" :WL.MNombre,
            "matWR" :WR.MNombre,

            "saT" : saT,
            "saP" : saP,
            "saWF" : saWF,
            "saWB" : saWB,
            "saWL" : saWL,
            "saWR" : saWR,

            "cT"  :T.MCoefAcu,
            "cP"  :P.MCoefAcu,
            "cWF" :WF.MCoefAcu,
            "cWB" :WB.MCoefAcu,
            "cWL" :WL.MCoefAcu,
            "cWR" :WR.MCoefAcu,

            "ssaW":ssaW.tolist(),
            "sobjsaW":sobjsaW,
            "Extras": Extras,

            "RT60f":RT60f,
            "dev":round(R.dev,3),

            "RT60":round(R.RT60,3)
    }
            
    return HttpResponse(json.dumps(datos), content_type='application/json')

class Resonador():
    def __init__(self,fr,q):
        self.fr = fr
        self.q  = q
        self.bw = fr/q
        self.fh = fr + self.bw/2
        self.fl = fr - self.bw/2
        self.RT60 = 2.2/self.bw
        
class Difragmatico(Resonador):
    def panelD(self, b, h, m):
        self.b = b
        self.h = h
        self.m = m
        self.d = m/(b*h)
        self.e = ((6/self.fr)**2)/self.d


class Perforado(Resonador):
    def panelP(self, b, h, g, d, p):
        self.b = b
        self.h = h
        self.g = g
        self.d = d
        self.p = p
        self.s = ((78.5*(2.54*d)**2)/p)**(1/2.0)
        self.nb = round((b/self.s)-1)
        self.nh = round((h/self.s)-1)
        self.n = self.nb*self.nh
        self.e = ((548/self.fr)**2)*(p/(g+0.8*(2.54*d)))
        self.borde_altura = (self.b - (round(self.s,2)*(self.nb-1)))/2
        self.borde_base = (self.h - (round(self.s,2)*(self.nh-1)))/2

def Paneles(request):
    clear()
    Paneles_Loader = loader.get_template('Paneles.html')
    Paneles_Template = Paneles_Loader.render({})
    return HttpResponse(Paneles_Template)

def Paneles_resp(request):
  
    if request.method == "GET":
        clear()      

        Tipo = request.GET.get("tipo")
        if Tipo == "Difragmático":
            P = Difragmatico(   float(request.GET.get("f")),
                                float(request.GET.get("q")))

            P.panelD(   float(request.GET.get("db")),
                        float(request.GET.get("dh")),
                        float(request.GET.get("dm"))
            )

            datos = {
                "f":P.fr,
                "q":P.q,

                "bw":round(P.bw,2),
                "fh":round(P.fh,2),
                "fl":round(P.fl,2),
                "RT60":round(P.RT60,3),

                "db":P.b,
                "dh":P.h,
                "dm":P.m,
                "dd":round(P.d*100000,2),
                "de":round(P.e,2)
            }

        if Tipo == "Perforado":
            P = Perforado(   float(request.GET.get("f")),
                            float(request.GET.get("q")))

            P.panelP(   float(request.GET.get("pb")),
                        float(request.GET.get("ph")),
                        float(request.GET.get("pg")),
                        float(request.GET.get("pd")),
                        float(request.GET.get("pp")),
            )

            print(P.borde_base)
            print(P.borde_altura)

            datos = {
                "f":P.fr,
                "q":P.q,
                "bw":round(P.bw,2),
                "fh":round(P.fh,2),
                "fl":round(P.fl,2),
                "RT60":round(P.RT60,3),

                "pb":P.b,
                "ph":P.h,
                "pg":P.g,
                "pd":P.d,
                "pp":P.p,
                "pnb":P.nb,
                "pnh":P.nh,

                "pn":round(P.n),
                "ps":round(P.s,2),
                "pe":round(P.e,2),

                "borde_base":round(P.borde_base,2),
                "borde_altura":round(P.borde_altura,2)
            }

    return HttpResponse(json.dumps(datos), content_type='application/json')

bw3 = 31.25
bw2 = bw3/2**(1/3.0)
bw1 = bw2/2**(1/3.0)

Tercios_Octava = np.array([bw1,bw2,bw3])
Tercios_Octava2 = np.array([bw1,bw2,bw3])
for i in range (2,6,1):
    Tercios_Octava2 = np.array(Tercios_Octava2*2)
    Tercios_Octava  = np.concatenate((Tercios_Octava,Tercios_Octava2),axis=None)
    
Tercios_Octava_Low  = Tercios_Octava/2**(1/6)
Tercios_Octava_High = Tercios_Octava*2**(1/6)

n = 20

class room_modos():
    def __init__ (self, x, y, z, RT60):
        self.x = x
        self.y = y
        self.z = z
        self.d = np.array([x,y,z])
        self.RT60 = RT60
        self.v = x*y*z
        self.max = max(x,y,z)
        self.f1	 = 343/(2*self.max)
        self.f2  = 1893*(self.RT60/self.v)**(1/2.0)
        self.f3	 = self.f2*4
        self.sf  = 2000*(self.RT60/self.v)**(1/2.0)
        self.bw  = 2.2/self.RT60

    def modo_axial(self):
        self.axial = np.ones((n,3))
        for i in range(1,n+1,1):
            x = 343/2*i/self.d
            self.axial[i-1] = x*self.axial[i-1]

        
    def conteo(self):
        self.conteo_modos = np.zeros((4,15))

        for s in range(0,3,1):
            for e in range(0,n,1):
                for i in range(0,15,1):
                    if self.axial[e,s] > Tercios_Octava_Low[i] and self.axial[e,s] < Tercios_Octava_High[i]:
                        self.conteo_modos[s,i] += 1
        for i in range(0,15,1):
            self.conteo_modos[3,i] = self.conteo_modos[0,i] + self.conteo_modos[1,i] + self.conteo_modos[2,i]

def Modos(request):
    clear()
    Modos_Loader = loader.get_template('Modos.html')
    Modos_Template = Modos_Loader.render({})
    return HttpResponse(Modos_Template)

def Modos_resp(request):
    clear()
    if request.method == "GET":

        R = room_modos (float(request.GET.get("x")),
                  float(request.GET.get("y")),
                  float(request.GET.get("z")),
                  float(request.GET.get("RT60"))
                  )

        R.modo_axial()
        

        axial = np.around(R.axial,1,None)

        axial_lista = axial.tolist()

        R.conteo()

        conteo = R.conteo_modos

        conteo_lista = conteo.tolist()

        header_a = np.array([['' ,'x','y','z']])
        axial_df = pd.DataFrame(data=R.axial,index=range(1,n+1,1),columns=header_a[0,1:])
        conteo_df = pd.DataFrame(data = R.conteo_modos, index = ['x','y','z','suma'] , columns = np.round(Tercios_Octava))

        axial_orden = R.axial.flatten()
        axial_orden = np.sort(axial_orden)
        axial_orden = np.around(axial_orden,2,None)
        axial_orden = axial_orden.tolist()

        print(axial_df)
        print(conteo_df)
        
        datos = {
            "x":R.x,
            "y":R.y,
            "z":R.z,
            "RT60":R.RT60,

            "f1":round(R.f1),
            "f2":round(R.f2),
            "f3":round(R.f3),
            "sf":round(R.sf,2),
            "bw":round(R.bw,2),

            "axial":axial_lista,
            "conteo":conteo_lista,
            "axial_orden":axial_orden
        }

    return HttpResponse(json.dumps(datos), content_type='application/json')