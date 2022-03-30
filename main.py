from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.lang import Builder
from skimage import io #imagenes de muestra
import numpy as np
from skimage.color import rgb2gray
import skfuzzy as fuzz
KV='''
<MyWidget>:   
    canvas:
        Color:
            rgba: (0,1,0,1)
    cols:2
    rows:3
    id:my_widget
    
    Image:
        id:image1
        source:""

    Image:
        id:image2
        source:""

    FileChooserIconView:
        id:filechooser
        on_selection:my_widget.selected(filechooser.selection)
    GridLayout:
        rows:5
        Button:
            text: "Ecualizacion Histograma"
            color: .5,.7,1,1
            font_size: 20
            #on_press: print("The Button Is Pressed")
            on_press: root.click_me(filechooser.selection)
        Button:
            text: "Multiumbral"
            color: .5,.7,1,1
            font_size: 20
            #on_press: print("The Button Is Pressed")
            on_press: root.Otro_Metodo(filechooser.selection)
        Button:
            id: Otsu1
            text: "Otsu"
            #background_normal: 'normal3.png'
            background_color: 1,0,0,1
            color: .5,.7,1,1
            font_size: 20
            #on_press: print("The Button Is Pressed")
            on_press: 
                root.Otsu(filechooser.selection)

        Button:
            text: "High-Boost"
            color: .2,1,1,1
            font_size: 20
            #on_press: print("The Button Is Pressed")
            on_press: root.Unsharp(filechooser.selection)
        Button:
            text: "Fuzzy"
            color: .2,.8,.8,1
            font_size: 20
            #on_press: print("The Button Is Pressed")
            on_press: root.Fuzzy(filechooser.selection)

'''

class MyWidget(GridLayout):
    Builder.load_string(KV)
    def selected(self, filename):
        try:
            self.ids.image1.source = filename[0]

        except:
            pass
    def click_me(self,filename):
        histo = [0.0]*256
        pro = [0.0]*256

        #gris = data.camera()
        #Ima = io.imread('C:/Users/moman/Documents/UPIITA/8vo Semestre_UPIITA/Imagenologia\PorRegion/Ultrasonido/ultra1.jpg')
        Ima = io.imread(self.ids.image1.source)
        gris = rgb2gray(Ima)*255.0
        filas= gris.shape[0]
        columnas = gris.shape[1]

        # for i in range(filas):
        #     for j in range(columnas):
        #         gris[i,j]=int(abs(gris[i,j]))


        salida = [0.0] * filas
        for i in range(filas):
            salida[i] = [0.0] * columnas 

        #salida = np.zeros((filas,columnas))
        for i in range(filas):
            for j in range(columnas):
                pixel = int(gris[i,j])
                histo[pixel] += 1
                pro[pixel]=histo[pixel]/(filas*columnas)

        ecualiza = [0.0]*256
        acumulado = 0
        for k in range(256):
            acumulado = pro[k] +acumulado
            ecualiza[k] = acumulado * 255.0
            
        for i in range(filas):
            for j in range(columnas):
                entrada = int(gris[i,j])
                salida[i][j] = int(ecualiza[entrada])
        io.imsave("ecualizada.png",np.uint8(salida))
        self.ids.image2.source = "ecualizada.png"
        self.ids.image2.reload()
        print("completado")
    def Otro_Metodo(self,filename):
        imagen = io.imread(self.ids.image1.source)
        gris = np.uint8(rgb2gray(imagen)*255.0)
        histo = np.zeros(256)

        filas= gris.shape[0]
        columnas = gris.shape[1]
        salida = np.zeros((filas,columnas))
        #histograma de la imagen
        for i in range(filas):
            for j in range(columnas):
                pixel = gris[i,j]
                histo[pixel] += 1
        Prob = histo/(filas*columnas)

        P = np.zeros(256)
        S = np.zeros(256)

        P[0]= Prob[0]
        S[0] = 0*Prob[0]

        for v in range(len(Prob)-1):
            P[v+1] = P[v]+Prob[v+1]         # Ec 22
            S[v+1] = S[v]+(v+1)*Prob[v+1]   # Ec 23

        PP = np.zeros((256,256))
        SS = np.zeros((256,256))
        HH = np.zeros((256,256))

        resta1 = np.zeros(len(Prob)+2)
        resta1[1:-1] = P
        resta2 = np.zeros(len(Prob)+2)
        resta2[1:-1] = S

        for u in range(256):
            for v in range(256):
                PP[u,v] = P[v]-resta1[u]+0.0001 # Ec 24
                SS[u,v] = S[v]-resta2[u] # Ec 25
                HH[u,v] = (SS[u,v]**2)/(PP[u,v])  # Ec 29
                
        U = 0;
        CLA = 3;
        L = 255;

        for t1 in range(0,L-(CLA-1),1):
            for t2 in range (t1+1,L-(CLA-2),1):
                r1 = HH[1,t1] + HH[t1+1,t2] + HH[t2+1,L]
                if (U<r1):
                    U = r1
                    umbral = np.array([t1,t2])-1
                    # Poner en cascada los ciclos para mas umbrales

        # aplicar la "binarizacion" del multiumbralizado a la imaen
        nuevo = np.zeros((filas,columnas))
        for i in range(filas):
            for j in range(columnas):
                k = gris[i,j]
                if k<=umbral[0]:
                    nuevo[i,j]=np.uint8(255/3)
                if k> umbral[0]:
                    if k<=umbral[1]:
                        nuevo[i,j]=np.uint8(2*255/3)
                    else:
                        nuevo[i,j]=np.uint8(255)
        io.imsave("ecualizada.png",nuevo)
        self.ids.image2.source = "ecualizada.png"
        self.ids.image2.reload()
    def Otsu(self,filename):
        self.ids.Otsu1.background_color =  1,1,0,1
        histo = np.zeros(256)
        gris = io.imread(self.ids.image1.source)
        gris = np.uint8(rgb2gray(gris)*255.0)
        filas= gris.shape[0]
        columnas = gris.shape[1]
        salida = np.zeros((filas,columnas))
        nuevo = np.zeros((filas,columnas))
        #histograma de la imagen
        for i in range(filas):
            for j in range(columnas):
                pixel = gris[i,j]
                histo[pixel] += 1
                
        pro = histo/(filas*columnas)
        prob = histo/(filas*columnas)
        suma1 = 0.000001 # para que no se divida entre 0 en la primera iteracion
        suma3 =  0
        omega0 = []
        omega1 = []
        mu0 = []
        mu1 = []
        for k in range(256):
            suma1 += prob[k]
            suma3 += k*prob[k]
            omega0.append(suma1) # ecuacion 2
            mu0.append(suma3/suma1) # ecuacion 4
            suma2 = 0.000001 # para que no se divida entre 0 en la primera iteracion
            suma4 = 0
            for j in range(k+1,256,1):
                suma2 += prob[j]
                suma4 += j*prob[j]
            omega1.append(suma2) # ecuacion 3 
            mu1.append(suma4/suma2) # ecuacion 5
        umbral = np.array(omega0)*np.array(omega1)*(np.array(mu1)-np.array(mu0))**2
        umb = np.where(umbral == np.amax(umbral))
        Umbr=umb[0][0]
        for i in range(filas):
            for j in range(columnas):
                k = gris[i,j]
                if k<=Umbr:
                    nuevo[i][j]=0
                else:
                    nuevo[i][j]=255
        io.imsave("ecualizada.png",np.uint8(nuevo))
        self.ids.image2.source = "ecualizada.png"
        self.ids.image2.reload()
        print("completado")
        self.ids.Otsu1.background_color =  1,0,0,1
    def Unsharp(self,filename):
        #plt.close('all')
        
        foto = io.imread(self.ids.image1.source)
        #foto = data.retina()
        [filas, columnas, capas] = foto.shape
        o=3
        A=1.1
        #o = int((int(input("Introduzca el orden de las vecindades:    "))-1)/2)
        #A = np.float64(input("Introduzca el valor de A:    "))

        mascara_pasaaltas=-1*np.ones((2*o+1,2*o+1))
        mascara_pasaaltas[o][o]=A*np.float64(np.power(2*o+1,2)-1)

        im_fil_pasaaltas = np.zeros((filas,columnas,capas))


        for i in range(o,filas-o-1):
            for j in range(o,columnas-o-1):
                mat_vecindad_r = foto[i-o:i+o+1,j-o:j+o+1,0]
                mat_vecindad_g = foto[i-o:i+o+1,j-o:j+o+1,1]
                mat_vecindad_b = foto[i-o:i+o+1,j-o:j+o+1,2]
                mat_vecindad_mod_r=mat_vecindad_r*mascara_pasaaltas
                mat_vecindad_mod_g=mat_vecindad_g*mascara_pasaaltas
                mat_vecindad_mod_b=mat_vecindad_b*mascara_pasaaltas
                im_fil_pasaaltas[i,j,0]=np.uint8(np.sum(mat_vecindad_mod_r/np.power(2*o+1,2)))
                im_fil_pasaaltas[i,j,1]=np.uint8(np.sum(mat_vecindad_mod_g/np.power(2*o+1,2)))
                im_fil_pasaaltas[i,j,2]=np.uint8(np.sum(mat_vecindad_mod_b/np.power(2*o+1,2)))        
        io.imsave("ecualizada.png",np.uint8(im_fil_pasaaltas))
        self.ids.image2.source = "ecualizada.png"
        self.ids.image2.reload()
        print("completado")
    def Fuzzy(self,filename):
        #Generar conjuntos difusos

        pixel = np.linspace(0, 255, 256)        # define que pixeles se varán afectados
        claros = fuzz.smf(pixel, 130, 230)
        grises = fuzz.gbellmf(pixel, 55, 3, 128)
        oscuros = fuzz.zmf(pixel, 25, 130) 
        #singletons para pasar de difuso a real
        s1 = 30                    # define que tan agresivo es el cambio
        s2 = 40
        s3 = 100

        #observar grafica de salida de ecualizacion
        salida = np.zeros(256)
        for i in range (256):
            salida [i] = ((oscuros[i]*s1)+(grises[i]*s2)+(claros[i]*s3)) / (oscuros[i]+grises[i]+claros[i])
        #=======================================================================#
        #Usar ecualizacion difusa
        Ima = io.imread(self.ids.image1.source)
        gris = np.uint8(rgb2gray(Ima)*255.0)
        #gris = data.camera()
        [filas, columnas] = gris.shape
        EHF = np.zeros( (filas, columnas) )

        for i in range(filas):
            for j in range(columnas):
                valor = gris[i, j]
                EHF[i,j] = np.uint8(salida[valor])
        io.imsave("ecualizada.png",np.uint8(EHF))
        self.ids.image2.source = "ecualizada.png"
        self.ids.image2.reload()
        print("completado")


class FileChooserWindow(App):
    def build(self):
        return MyWidget()

if __name__ == "__main__":
    window = FileChooserWindow()
    window.run()

