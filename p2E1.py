# -*- coding: utf-8 -*-
"""
En el siguiente código se van a implementar las clases y funciones 
    necesarias para generar las gráficas correspondientes al campo B
    generado por una espira (Biot y Savart), por un dipolo magnético y 
    por una distribución de estos. 
    
Para cada uno de los casos es necesario variar alguno de los parámetros
    para que el resultado obtenido tenga una forma aceptable y pueda
    ser comparado con los demás. Estos se especifican a lo largo del código.
    
Ante cualquier duda consultar la documentación de cada clase y/o función,
    así como los comentarios que hay dentro de estas, donde se detalla
    el procedimiento utilizado y se explican aquellos pasos de mas 
    difícil comprensión.
    

@author: Guillem Pellicer Chover
"""

# =============================================================================
# Importamos los módulos necesarios
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import sys as sys
from numba import jit
from mpl_toolkits.mplot3d import Axes3D # Aunque parezca que no se use es
                                            # indispensable para los gráficos 
                                            # 3D ya que sin este módulo el 
                                            # parámetro projection = '3d' 
                                            # no funciona

# =============================================================================
# Definimos las funciones de cálculo vectorial en una clase de operaciones
# =============================================================================

class Operaciones:
    
    '''
    Clase de operadores matemáticos, en la cual se definen un conjunto
        de funciones necesarias para el buen funcionamiento del código, que
        van a servir para hacer mas intuitivo y comprensible el procedimiento,
        facilitando su legibilidad y haciendo que este se asemeje lo más 
        posible a las ecuaciones que se utilizan.
        
    Se presupone que los parámetros introducidos en las siguientes funciones
        tienen las dimensiones y son del tipo correcto. No se consideran las 
        excepciones en caso de no ser así. No respetar esto puede producir 
        que el programa no funcione bien o que incluso arroje un error 
        y se interrumpa.
    '''
    
    def __init__(self):
        pass

    @jit(cache=True)   
    def producto_exterior(self, a, u):
        return [a*u[i] for i in range(len(u))]
    
    @jit(cache=True)        
    def producto_escalar(self, u, v):
        producto = 0
        for i in range(min(len(u),len(v))): producto += np.array(u[i]*v[i])
        return producto
    
    @jit(cache=True)    
    def producto_vectorial(self, u, v):
        return [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
    
    @jit(cache=True)    
    def suma_vectores(self, u, v):
        return [u[i]+v[i] for i in range(len(u))]
    
    @jit(cache=True)    
    def resta_vectores(self, u, v):
        return [u[i]-v[i] for i in range(len(u))]
    
    @jit(cache=True)    
    def norma_vector(self, u):
        return np.sqrt(u[0]**2 +u[1]**2 + u[2]**2)
 
    
ops = Operaciones() # llamamos a la clase como ops


# =============================================================================
# Definimos las clases en las que se realizaran los cálculos de B
# =============================================================================

class Biot_Savart_V0:
    
    ''' 
    Clase que necesita un parametro n (subdivisiones de la matriz),así como
        la intensidad y el radio de la espira para funcionar.
    
    Basada en dos funciones:
        
        dB, que calcula el campo magnético (diferencial de campo) producido 
            por una espira para un elemento diferencial de longitud
        
        y
        
        B, que generaliza el valor diferencial de campo (obtenido por dB)
            para todo el espacio objeto de estudio.
    '''
    
    def __init__(self,n,I,R):
        
        self.n = n
        self.I = I
        self.R = R

    @jit(cache=True)
    def dB(self,dl,d):
        
        '''
        Aquí se calcula, con los parámetros I (intensidad), r (vector 
            distancia) y dl (elemento diferencial de longitud), el valor
            del campo magnético generado por el elemento dl a una distancia 
            r de este.
        '''
        
        
        modr = ops.norma_vector(d)  # Norma del vector distancia para cada 
                                        # punto del espacio
        
        '''Cabe destacar que :
                Mu_0 = 4pi*10e-7
           y que:
                c = mu_0/4pi·I/r^3'''
           
        c = 1/modr**3               # Escalar que multiplica a B para cada 
                                        # punto y que varía en función de
                                        # este

        
        return c*ops.producto_vectorial(dl,d)
    
    @jit(cache=True)
    def B(self,r,dB):
        
        '''
        Finalmente, mediante los valores proporcionados por la función
            dB(), calculamos el valor de B para cada punto del espacio.
        
        Inicializamos las componentes del Campo B.
        
        Definimos el valor del diferencial de ángulo (da).
        
        Definimos la constante k como el escalar que múltiplica a cada
            componente de B y que no ha sido considerada anteriormente 
            en la función dB().
        
        Mediante un bucle for, calculamos el valor de B para cada punto
            del espacio.
        '''
        
        B = [0,0,0]                # Valores iniciales para B
        da = 2*np.pi/self.n        # Una vuelta entre la resolución
        k = 1e-7*self.I**2*self.R  # Por legibilidad del código, escalar que
                                       # multiplica a cada componente de B 

        for i in range(self.n+1):
            
            '''
            Con las variables antes inicializadas, definimos el diferencial
                de longitud (dl), en función de funciones trigonométricas.
                
            Hacemos lo propio con el vector distancia.
            
            Procedemos a calcular B para este punto, a multiplicar este por
                las constantes necesarias y a introducir dichos valores
                en la lista inicializada, antes de salir del bucle y repetir
                el proceso para un punto dostinto.
            '''
            
            dl = (-self.R*np.sin(i*da),self.R*np.cos(i*da),0)
            distancia = (self.R*np.cos(i*da),self.R*np.sin(i*da),0)
            d  = ops.resta_vectores(r, distancia)
            Bi = k/(ops.norma_vector(d)**3)*dB(dl,d)
            
            B[0] += Bi[0]
            B[1] += Bi[1]
            B[2] += Bi[2]
        
        ''' Finalmente devolvemos los valores deseados'''
        
        return B[0], B[1], B[2]
    
class Biot_Savart:
    
    ''' 
                ¡¡¡VERSION MEJORADA DE LA CLASE ANTERIOR!!!
    
    Clase que necesita un parametro n (subdivisiones de la matriz),así como
        la intensidad y el radio de la espira para funcionar.
    
    Basada en una única funcion, la cual recoge en una las dos utilizadas
    en la clase anterior:
    
        B, que calcula el valor del campo magnético para cada punto del 
        espacio, utilizando una generalización y adaptación angular
        de la ley de Biot-Savart.
    '''
    
    def __init__(self,n,I,R):
        
        self.n = n
        self.I = I
        self.R = R
    
    @jit(cache=True)
    def B(self,r):
        
        '''
        Inicializamos las componentes del Campo B. 
        
        Definimos el valor del diferencial de ángulo (da).
        
        Definimos el valor de la constante que multiplicará a cada
            componente del campo, dejando de lado el módulo del vector
            r, que varía para cada punto y que será añadido después.
        
        Mediante un bucle for, calculamos el valor de B para cada punto
            del espacio.
        '''
        
        B = [0,0,0]               # Valores iniciales para B
        da = 2*np.pi/self.n       # Una vuelta entre la resolución
        
        '''Cabe destacar que Mu_0=4pi*10e-7'''
        
        k = 1e-7*self.I**2*self.R # Por legibilidad del código

        for i in range(self.n+1):
            
            '''
            Con las variables antes inicializadas, definimos el diferencial
                de longitud (dl), en función de funciones trigonométricas.
                
            Hacemos lo propio con el vector distancia.
            '''
    
            dl = (-self.R*np.sin(i*da),self.R*np.cos(i*da),0)
            distancia = (self.R*np.cos(i*da),self.R*np.sin(i*da),0)
            
            '''
            Aquí realizamos los cálculos que anteriormente se realizaban en
                la función dB().
                
            Primero calculamos el vector d (análogo al r de la fórmula) y
                su módulo.
                
            Luego múltiplicamos la constante k dividida por dicho módulo al
                producto escalar de r (d) y el vector diferencial de 
                longitud (dl), reproduciendo así la fórmula de la ley 
                de Biot-Savart.
            
            Finalmente sumamos cada una de las componentes del campo a la 
                matriz B previamente inicializada.
            '''
            
            d  = ops.resta_vectores(r,distancia)
            modr = ops.norma_vector(d)**3
            Bi = k/modr*ops.producto_vectorial(dl,d)
        
            
            B[0] += Bi[0]
            B[1] += Bi[1]
            B[2] += Bi[2]
            
        ''' Finalmente devolvemos los valores deseados'''
        
        return B[0], B[1], B[2]
    
class MDipolar:
    
    '''
    Clase que implementa una función B para calcular el campo magnético
        producido por un momento dipolar.
    
    Necesita 4 parámetros para funcionar:
        
         n = resolución de la espira
         R = radio de la espira
         I = intensidad que circula por esta
         a = forma normalizada del vector momento dipolar
         
    '''
    
    def __init__(self,n,R,I,a):
        
        self.n = n
        self.R = R
        self.I = I
        self.a = a
        
   
    @jit(cache=True)
    def B(self,r):
        
        '''
        Primeramente creamos una matriz de modulos de distancias R, para cada
            punto del espacio. 
            
        Luego definimos la constante c = mu_0/4pi
        '''
        
        R = ops.norma_vector(r)
        c = 1e-7
        
        '''
        Ahora definimos el vector m mediante la relación de su módulo
            con I, R y pi. Una vez hecho esto lo multiplicamos por el
            vector m normalizado que es necesario introducir como 
            parámetro, obteniendo asi el vector momento dipolar deseado.
        
        Después defnimos el producto m·n, denotado como p, haciendo uso de la
            función producto escalar, definida en MathOps.
        '''
        
        m = ops.producto_exterior(self.I*np.pi*self.R**2,self.a)
        p = ops.producto_escalar(m,r)
        
        '''
        Para terminar, haciendo uso de la ecuación deducida teóricamente, y 
            mediante los vectores y constantes antes calculadas, obtenemos
            las componentes del campo B
        '''
        
        
        Bx = c*(-(3*p)*r[0] + (R)**2*m[0])/R**3
        By = c*(-(3*p)*r[1] + (R)**2*m[1])/R**3
        Bz = c*(-(3*p)*r[2] + (R)**2*m[2])/R**3
        
        ''' Finalmente devolvemos los valores deseados'''
        
        return Bx, By, Bz
    
    
class Dist_Cuad_MDipolar:
    
    '''
    Clase que generaliza la anterior, generando una distribución cuadrada de 
        momento dipolares y calculando el campo producido por esta.
        
    Para funcionar esta clase necesita cinco parametros:
        
        n    = resolución de la espira
        m    = vector momento angular normalizado
        R    = radio de la espira
        I    = intensidad que circula por esta
        func = parámetro entero que determina cual de los dos procedimientos
                    para el cálculo de B se va a utilizar. En caso de ser
                    distinto a 1 o 2, el script interrumpe su ejecución.
    '''

    
    def __init__(self,n,m,I,R,func):
        
        self.n    = n
        self.m    = m
        self.R    = R
        self.I    = I
        self.func = func
    
    @jit(cache=True)    
    def B(self, X, Y, Z):
            
        '''
        Consideramos el parámetro func. Si se ha introducido 
        un valor adecuado, procedemos con el cálculo deseado.
        '''
            
            
        if self.func == 1:
                
            '''
            En un inicio, definimos la constante c = mu_0/4pi
            Para este caso inicializamos el vector B, formado por tres
                matrices llenas de 0s. La dimensión de las matrices 
                viene dada por el valor n, introducido por parámetro a
                la clase.
                
            Luego expresamos m en funcion de R, I y pi
            '''
            c = 1e-7    
            B = np.zeros((3,self.n,self.n))
            
            '''
            En el caso de querer utilizar m = m(I,R) hay que cambiar
            los límites del graficado, multiplicando sus límites superior
            e inferior por el módulo de m
            '''
                
            # self.m = ops.producto_exterior(self.I*np.pi*self.R**2,self.m)
                
            '''
            Ahora, con los dos primeros bucles, recorremos cada una de
                las matrices del vector B.
            '''
            for x in range(0,self.n):
                for y in range(0,self.n):
                        
                    '''
                    Ahora, para cada elemento de la matriz, calculamos 
                        el campo producido por cada dipolo y los sumamos
                    '''
                        
                    for j in range(0,self.n):
                            
                        '''
                        Definimos el vector distancia, dependiente de
                            cada dipolo, calculamos su módulo y también
                            el producto m·n (m·r).
                                
                        Luego mediante la fórmula del campo, calculamos 
                            cada componente y las sumamos a las
                            calculadas anteriormente
                        '''
                                
                        r = X[y][j],Y[y][j],Z[x][y]
                        R = ops.norma_vector(r)
                        p = ops.producto_escalar(self.m,r)
                                    
                        B[0][x][y] += c*(m[0]-3*(p)*X[y][j])/R**3
                        B[1][x][y] += c*(m[1]-3*(p)*Y[y][j])/R**3
                        B[2][x][y] += c*(m[2]-3*(p)*Z[y][j])/R**3
                     
            
        elif self.func == 2: 
                
            '''
            En un inicio, definimos la constante c = mu_0/4pi
            Para empezar, inicializamos la variable B, así como un 
                vector de posición de momentos iniciales y una lista vacía 
                para actualizar nuestra distribución.
                
            Luego expresamos m en función de I, R y pi.
            '''
            c = 1e-7    
            B = [0,0,0]
            moment = np.linspace(-10,10,self.n)
            distribucion = []
            
            '''
            En el caso de querer utilizar m = m(I,R) hay que cambiar
            los límites del graficado, multiplicando sus límites superior
            e inferior por el módulo de m
            '''
            
            # self.m = ops.producto_exterior(self.I*np.pi*self.R**2,self.m)
                
            '''
            Ahora rellenamos nuestra matriz de momentos a t0 actualizando
                la lista distribución con los elementos del vector
                moment (posición de momentos a t0).
            '''
                
            for x in range(0,len(moment)):
                for y in range(0,len(moment)):
                    distribucion.append([moment[x],moment[y]])
                
            for j in range(0,len(distribucion)):
                    
                '''
                Aquí definimos las variables necesarias, distancia,
                    mom (momento), p (m·n) y R (norma de r)
                '''
                    
                distancia = ops.resta_vectores([X,Y,Z],distribucion[j]+[0]) 
                mom=ops.producto_exterior(
                    1/ops.norma_vector(distancia),distancia) 
                p = ops.producto_escalar(self.m,mom)
                R = ops.norma_vector([X,Y,Z])
                    
                '''
                Para terminar actualizamos el vector B, anteriormente
                    inicializado.
                '''
                    
                B[0] += (c/R**3)*(m[0]-3*(p)*mom[0])
                B[1] += (c/R**3)*(m[1]-3*(p)*mom[1])
                B[2] += (c/R**3)*(m[2]-3*(p)*mom[2])
            
        else:
                
            '''
            En caso de que func tenga un valor no contemplado, 
                se imprime un mensaje de error y se interrumpe la ejecución
            '''
                
            print('Elige una función existente...')
            sys.exit()

        '''Finalmente devolvemos las componentes de B'''

        return B[0], B[1], B[2]


# =============================================================================
# Declaramos las variables que vamos a utilizar (variables = n, R, I, m)
# =============================================================================


variablesBSV1 = 100, 0.5  , 100, [1,0,1] 
variablesBSV2 = 100, 0.005, 100, [1,0,1]
variablesBS3D = 20 , 0.5  , 100, [1,0,1]
variablesMDIP = 100, 0.5  , 100, [0,0,1]
variablesMD3D = 12 , 0.5  , 100, [0,0,-1]
variablesDMD1 = 10 , 0.5  , 100, [1,0,1]
variablesDMD2 = 10 , 0.5  , 100, [0,0,1]


# =============================================================================
# Procedemos con el cálculo
# =============================================================================



'''Biot-Savart R = 0.5'''



# Aquí llamamos a las variables, generamos la malla y realizamos el cálculo

n, R, I, m = variablesBSV1 # Llamamos a las variables

campo = Biot_Savart(n,I,R) # Creamos un objeto llamado campo

x, z = np.linspace(-4,4,n), np.linspace(-4,4,n)  # Creamos los vectores                                                   
X, Z = np.meshgrid(x,z) # Generamos la malla con los vectores

r = [X,0,Z] # Creamos el vector r a utilizar

Bx, By, Bz = campo.B(r) # Calculamos las componentes de B


# Ahora realizamos el proceso de graficado

fig, ax = plt.subplots(1, 1) # Creamos la figura y los ejes

color = 2 * np.log(np.hypot(Bx, Bz)) # Calculamos el color a utilizar 


# Dibujamos el resultado de campo.B deseado
ax.streamplot(X, Z, Bx, Bz, color = color,
              linewidth = 1, cmap = plt.cm.inferno, density=2.5,
              arrowstyle = '->', arrowsize = 1.5) # Importante el density!!

# ax.set_title("Campo Magnético") # Ponemos título a la figura
ax.set_xlim(-4,4) # Delimitamos la figura
ax.set_ylim(-4,4) # Delimitamos la figura

plt.show() # Mostramos el resultado

x, z = np.linspace(0,2,100), np.linspace(0,2,100) # Creamos los vectores 
X, Z = np.meshgrid(x, z) # Generamos la malla con los vectores


# Con 0 en vez de zeros la ejecución da problemas
zeros = np.zeros((n,n))

r = (zeros,zeros,Z) # Creamos el vector r a utilizar
Bx, By, Bz = campo.B(r) # Calculamos las componentes de B


fig = plt.figure()
plt.plot(z,Bz)

del campo # Eliminamos el objeto campo



'''Biot-Savart R = 0.005'''



# Aquí llamamos a las variables, generamos la malla y realizamos el cálculo

n, R, I, m = variablesBSV2 # Llamamos a las variables

campo = Biot_Savart(n,I,R) # Creamos un objeto llamado campo

x, z = np.linspace(-4,4,n), np.linspace(-4,4,n)  # Creamos los vectores                                                   
X, Z = np.meshgrid(x,z) # Generamos la malla con los vectores

r = [X,0,Z] # Creamos el vector r a utilizar

Bx, By, Bz = campo.B(r) # Calculamos las componentes de B


# Ahora realizamos el proceso de graficado

fig, ax = plt.subplots(1, 1) # Creamos la figura y los ejes

color = 2 * np.log(np.hypot(Bx, Bz)) # Calculamos el color a utilizar 

# Dibujamos el resultado de campo.B deseado
ax.streamplot(X, Z, Bx, Bz, color = color,
              linewidth = 1, cmap = plt.cm.inferno, density=2.5,
              arrowstyle = '->', arrowsize = 1.5) # Importante el density!!

# ax.set_title("Campo Magnético") # Ponemos título a la figura
ax.set_xlim(-4,4) # Delimitamos la figura
ax.set_ylim(-4,4) # Delimitamos la figura

plt.show() # Mostramos el resultado

x, z = np.linspace(0,2,100), np.linspace(0,2,100) # Creamos los vectores 
X, Z = np.meshgrid(x, z) # Generamos la malla con los vectores

zeros = np.zeros((n,n))

r = (zeros,zeros,Z) # Creamos el vector r a utilizar
Bx, By, Bz = campo.B(r) # Calculamos las componentes de B


fig = plt.figure()
plt.plot(z,Bz)


del campo # Eliminamos el objeto campo



'''Momento Dipolar'''



# Aquí llamamos a las variables, generamos la malla y realizamos el cálculo

n, R, I, m = variablesMDIP # Llamamos a las variables

campo = MDipolar(n,R,I,m) # Creamos un objeto llamado campo

x, z = np.linspace(-4,4,n), np.linspace(-4,4,n) # Creamos los vectores 
X,Z = np.meshgrid(x,z) # Generamos la malla con los vectores


r = [X,0,Z] # Creamos el vector r a utilizar

Bx,By,Bz = campo.B(r) # Calculamos las componentes de B

# Ahora realizamos el proceso de graficado

fig, ax= plt.subplots(1,1) # Creamos la figura y los ejes

color = 2*np.log(np.hypot(Bx, Bz)) # Calculamos el color a utilizar

# Dibujamos el resultado de campo.B deseado
ax.streamplot(X, Z, Bx, Bz, color=color, linewidth=1, cmap=plt.cm.inferno,
              density=2, arrowstyle='->', arrowsize=1.5)

# ax.set_title("Campo Magnético") # Ponemos título a la figura
ax.set_xlim(-4,4) # Delimitamos la figura
ax.set_ylim(-4,4) # Delimitamos la figura

plt.show() # Mostramos el resultado


x, z = np.linspace(0,2,100), np.linspace(0,2,100) # Creamos los vectores 
X, Z = np.meshgrid(x, z) # Generamos la malla con los vectores

r = (0,0,Z) # Creamos el vector r a utilizar

Bx, By, Bz = campo.B(r) # Calculamos las componentes de B


fig = plt.figure()
plt.plot(z,np.abs(Bz))



del campo # Eliminamos el objeto campo



'''Momento Dipolar 3D'''



n, R, I, m = variablesMD3D # Llamamos a las variables

campo = MDipolar(n,R,I,m) # Creamos un objeto llamado campo

# Creamos los vectores 
x, y, z = np.linspace(-4,4,n), np.linspace(-4,4,n), np.array([-4,0,4]) 
X, Y, Z = np.meshgrid(x,y,z) # Generamos la malla con los vectores

r = [X,Y,Z] # Creamos el vector r a utilizar

Bx,By,Bz = campo.B(r) # Calculamos las componentes de B

# Ahora realizamos el proceso de graficado

fig = plt.figure() # Creamos la figura
ax = fig.add_subplot(111, projection='3d') # Creamos los ejes

# Dibujamos el resultado de campo.B deseado
ax.quiver(X, Y, Z, Bx, By, Bz ,length=1, color = 'b',  normalize=True)

# ax.set_title("Campo Magnético") # Ponemos título a la figura
plt.show() # Mostramos el resultado

del campo # Eliminamos el objeto campo



'''Distribución de Momentos Dipolares, método 1'''



n, R, I, m = variablesDMD1 # Llamamos a las variables
func = 1 # Damos valor a func

campo = Dist_Cuad_MDipolar(n,m,I,R,func)  # Creamos un objeto llamado campo



# Creamos los vectores 
x, y, z = np.linspace(-2, 2, n), np.linspace(-2, 2, n), np.linspace(-2, 2, n)
# Generamos la malla con los vectores
X, Y, Z = np.meshgrid(x, y)[0], np.meshgrid(x, y)[1], np.meshgrid(x, z)[1]

zeros = np.zeros(np.shape(X)) # Creamos una matriz de ceros (para evitar
                              # problemas)

Bx, By, Bz = campo.B(X,Y,Z) # Calculamos las componentes de B

fig, ax= plt.subplots(1,1) # Creamos la figura y los ejes

color = 2 * np.log(np.hypot(Bx, Bz)) # Calculamos el color a utilizar

# Dibujamos el resultado de campo.B deseado
ax.streamplot(X, Z, Bx, Bz, color=color, linewidth=1, cmap=plt.cm.inferno, 
              density=2, arrowstyle='->', arrowsize=1.5)
# plt.title("Campo Magnético")# Ponemos título a la figura
ax.set_xlabel("x") # Definimos la etiqueta del eje x
ax.set_ylabel("z") # Definimos la etiqueta del eje y
ax.set_xlim(-2,2) # Delimitamos la figura
ax.set_ylim(-2,2) # Delimitamos la figura


# Elegimos el aspecto de los ejes
plt.gca().set_aspect('equal', adjustable='box')
plt.show()# Mostramos el resultado


fig, ax= plt.subplots(1,1) # Creamos la figura y los ejes

color = 2 * np.log(np.hypot(Bx, By)) # Calculamos el color a utilizar

# Dibujamos el resultado de campo.B deseado
ax.streamplot(X, Y, Bx, By, color=color, linewidth=1, cmap=plt.cm.inferno,
              density=2, arrowstyle='->', arrowsize=1.5)

# plt.title("Campo Magnético")# Ponemos título a la figura
ax.set_xlabel("x") # Definimos la etiqueta del eje x
ax.set_ylabel("y") # Definimos la etiqueta del eje y
ax.set_xlim(-2,2) # Delimitamos la figura
ax.set_ylim(-2,2) # Delimitamos la figura

# Elegimos el aspecto de los ejes
plt.gca().set_aspect('equal', adjustable='box')
plt.show()# Mostramos el resultado

del campo # Eliminamos el objeto campo




'''Distribución de Momentos Dipolares, método 2'''



n, R, I, m = variablesDMD2 # Llamamos a las variables
func = 2 # Damos valor a func

campo = Dist_Cuad_MDipolar(n,m,I,R,func)  # Creamos un objeto llamado campo



# Creamos los vectores 
x, y, z = np.linspace(-20, 20, n), np.linspace(
                        -20, 20, n), np.linspace(-20, 20, n)
# Generamos la malla con los vectores
X, Y, Z = np.meshgrid(x, y)[0], np.meshgrid(x, y)[1], np.meshgrid(x, z)[1]

zeros = np.zeros(np.shape(X)) # Creamos una matriz de ceros (para evitar
                              # problemas)

Bx, By, Bz = campo.B(X,Y,Z) # Calculamos las componentes de B

fig, ax= plt.subplots(1,1) # Creamos la figura y los ejes

color = 2 * np.log(np.hypot(Bx, Bz)) # Calculamos el color a utilizar

# Dibujamos el resultado de campo.B deseado
ax.streamplot(X, Z, Bx, Bz, color=color, linewidth=1, cmap=plt.cm.inferno, 
              density=2, arrowstyle='->', arrowsize=1.5)
# plt.title("Campo Magnético")# Ponemos título a la figura
ax.set_xlabel("x") # Definimos la etiqueta del eje x
ax.set_ylabel("z") # Definimos la etiqueta del eje y
ax.set_xlim(-20,20) # Delimitamos la figura
ax.set_ylim(-20,20) # Delimitamos la figura


# Elegimos el aspecto de los ejes
plt.gca().set_aspect('equal', adjustable='box')
plt.show()# Mostramos el resultado


fig, ax= plt.subplots(1,1) # Creamos la figura y los ejes

color = 2 * np.log(np.hypot(Bx, By)) # Calculamos el color a utilizar

# Dibujamos el resultado de campo.B deseado
ax.streamplot(X, Y, Bx, By, color=color, linewidth=1, cmap=plt.cm.inferno, 
              density=2, arrowstyle='->', arrowsize=1.5)
# plt.title("Campo Magnético")# Ponemos título a la figura
ax.set_xlabel("x") # Definimos la etiqueta del eje x
ax.set_ylabel("y") # Definimos la etiqueta del eje y
ax.set_xlim(-20,20) # Delimitamos la figura
ax.set_ylim(-20,20) # Delimitamos la figura


# Elegimos el aspecto de los ejes
plt.gca().set_aspect('equal', adjustable='box')
plt.show()# Mostramos el resultado

del campo # Eliminamos el objeto campo



'''Distribución de Momentos Dipolares, método 2 (Variación en función de R)'''



n, R, I, m = variablesDMD2 # Llamamos a las variables
func = 2 # Damos valor a func

campo = Dist_Cuad_MDipolar(n,m,I,R,func)  # Creamos un objeto llamado campo


x, z = np.linspace(0,2,100), np.linspace(0,2,100) # Creamos los vectores 
X, Z = np.meshgrid(x, z) # Generamos la malla con los vectores

Bx, By, Bz = campo.B(0,0,Z) # Calculamos las componentes de B


fig = plt.figure()
plt.plot(z,Bz)
plt.ylim(0,0.0008)

x, z = np.linspace(1,2,100), np.linspace(1,2,100) # Creamos los vectores 
X, Z = np.meshgrid(x, z) # Generamos la malla con los vectores

Bx, By, Bz = campo.B(0,0,Z) # Calculamos las componentes de B


fig = plt.figure()
plt.plot(z,Bz)

x, z = np.linspace(1,5,100), np.linspace(1,5,100) # Creamos los vectores 
X, Z = np.meshgrid(x, z) # Generamos la malla con los vectores

Bx, By, Bz = campo.B(0,0,Z) # Calculamos las componentes de B


fig = plt.figure()
plt.plot(z,Bz)


del campo # Eliminamos el objeto campo

