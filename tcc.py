import numpy as NP
import matplotlib.pyplot as plt
import math
from scipy import linalg as LA



#delta de dirac

def delta(x,y):
	if (x==y):
		return 1
	else:
 		return 0

#Arredonda

def trunca(e_vals):
	for i in range(len(e_vals)):
		e_vals[i]=(round(e_vals[i],5))	
	return e_vals
def trunca2(e_vecs):
	for i in range(len(e_vecs[0])):
		for j in range(len(e_vecs[0])):
			e_vecs[i][j]=round(e_vecs[i][j],5)
	return e_vecs

#probabilidade classica

def probclas(e_vecs,e_vals,k,j,t):
	p=0
	for i in range(len(e_vals)):
		p=p+math.exp(-t*e_vals[i].real)*(NP.dot(k,e_vecs[i])*NP.dot(e_vecs[i],j))
	return p
#probabilidade quantica

def probquan(e_vecs,e_vals,k,j,t):
	p=0j
	for i in range(len(e_vals)):
		p=p+NP.cos(-t*e_vals[i])*(NP.dot(k,e_vecs[i])*NP.dot(e_vecs[i],j))+NP.sin(-t*e_vals[i])*(NP.dot(k,e_vecs[i])*NP.dot(e_vecs[i],j))*1j
	return (pow(p.real,2)+pow(p.imag,2))

#ineficiencia

def ineficiencia(e_vecs,e_vals,k,j,t):
	p=0
	for i in range(len(e_vals)):
		for l in range(len(e_vals)):
			p=p+delta(e_vals[i],e_vals[l])*(NP.dot(k,e_vecs[i])*NP.dot(e_vecs[i],j)*NP.dot(k,e_vecs[l])*NP.dot(e_vecs[l],j))
	return p

#Base canonica para matrizes de grau n

def basecanonica(n):
	C = NP.zeros((n,n),dtype=float)
	for i in range(n):
			C[i][i]=1
	return C

#Matriz do grau dos nos em uma rede anel

def mtgrau(n):
	D = NP.zeros((n,n),dtype=float)
	for i in range(n):
			D[i][i]=2
	return D
#matriz de adjacência em rede anel

def mtadj(n):
	A = NP.zeros((n,n),dtype=float)
	A[0][1]=1
	A[0][n-1]=1
	A[n-1][0]=1
	A[n-1][n-2]=1
	for i in range(1,n-1):
			A[i][i+1]=1
			A[i][i-1]=1
	return A
#media da probabilidade classica
def probclassmedia(e_vecs,e_vals,C,t):
	pm=0
	for i in range(len(e_vals)):
		pm=pm+probclas(e_vecs,e_vals,C[i],C[i],t)
	return pm/(len(e_vals))
#media da probabilidade quantica
def probquanmd(e_vecs,e_vals,C,t):
	pm=0
	for i in range(len(e_vals)):
		pm=pm+probquan(e_vecs,e_vals,C[i],C[i],t)
	return pm/(len(e_vals))
#media da ineficiencia
def probimd(e_vecs,e_vals,C,t):
	pm=0
	for i in range(len(e_vals)):
		pm=pm+ineficiencia(e_vecs,e_vals,C[i],C[i],t)
	return pm/(len(e_vals))


#dados para grafico probabilidade classica comecar num ponto e depois de certo tempo permanecer(ou voltar)
def yclassico(e_vecs,e_vals,C,tempo):
	Y = NP.zeros(len(tempo),dtype=float)
	for t in range(len(tempo)):
		Y[t]=probclassmedia(e_vecs,e_vals,C,t)
	return Y

# idem quantica
def yquantico(e_vecs,e_vals,C,tempo):
	Y = NP.zeros(len(tempo),dtype=float)
	for t in range(len(tempo)):
		Y[t]=probquanmd(e_vecs,e_vals,C,t)
	return Y
#idem ineficiencia
def yine(e_vecs,e_vals,C,tempo):
	Y = NP.zeros(len(tempo),dtype=float)
	for t in range(len(tempo)):
		Y[t]=probimd(e_vecs,e_vals,C,t)
	return Y


#teorico para rede triangular
def ytq(e_vecs,e_vals,C,tempo):
	Y = NP.zeros(len(tempo),dtype=float)
	for i in range(len(tempo)):
		Y[i]=(1/9)*(5+4*math.cos(3*tempo[i]))
	return Y
		



#dimensao da matriz
d=3
#base canonica

C = basecanonica(d)


#Matriz de adjacencia para rede triangular

A = mtadj(d)

#Matriz dos do grau de cada no

D = mtgrau(d)

#Matriz Laplaciano

L=D-A

e_vals, e_vecs=LA.eig(L)
#e_vals=trunca(e_vals)
#e_vecs=trunca2(e_vecs)
e_vecs=e_vecs.transpose()

#Probabilidade de iniciar em um ponto em um longo tempo permanecer neste ponto
#quantica
#print(probquan(e_vecs,e_vals,C[0],C[0],0))
#ineficiencia
#print(ineficiencia(e_vecs,e_vals,C[0],C[0],0))



tempo = NP.arange(0,0.5,0.01, dtype=float)

y1=yclassico(e_vecs,e_vals,C,tempo)
plt.title(u'Probabilidade clássica rede cíclica de '+str(d)+' nós')

plt.xlabel('tempo(s)')
plt.ylabel('Probabilidade')
plt.grid(True)
plt.plot(tempo,y1,'k-.')

plt.savefig('gclassico'+str(d))
plt.cla()

plt.title(u'Probabilidade quântica rede cíclica de '+str(d)+' nós')
plt.xlabel('tempo(s)')
plt.grid(True)

y2=yquantico(e_vecs,e_vals,C,tempo)
y4=ytq(e_vecs,e_vals,C,tempo)
plt.plot(tempo,y2,'b:s')
plt.plot(tempo,y4,'k:s')

plt.savefig('gquantico'+str(d))

plt.cla()

plt.title(u'Ineficiência na rede cíclica de '+str(d)+' nós')
plt.xlabel('tempo(s)')
plt.grid(True)

y3=yine(e_vecs,e_vals,C,tempo)

plt.plot(tempo,y3,'g:o')
plt.savefig('ginefic'+str(d))






