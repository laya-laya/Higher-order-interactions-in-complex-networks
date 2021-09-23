import numpy as np
from jitcsde import t,y as ys, jitcsde
import symengine
import matplotlib.pyplot as plt
from jitcode import jitcode, y
from tempfile import TemporaryFile


N = 6

int_freqs = np.random.uniform(0.8,1.0,N)
#couple_const1 = -0.22
K_2 = 0.02
K_3 = 0.01

# get adjacency matrix 
"""A = np.random.rand(N,N); A = A>0.7
B = np.random.rand(N,N,N); B = B>0.7
C = np.random.rand(N,N,N,N).reshape(N,N,N,N); C = C>0.7


np.savetxt("/Users/layaparkavousi/Desktop/A1.txt",A)
np.savetxt("/Users/layaparkavousi/Desktop/B1.txt",B.reshape(6,36))
np.savetxt("/Users/layaparkavousi/Desktop/C1.txt",C.reshape(6,216))"""

A = np.genfromtxt("/Users/layaparkavousi/Desktop/A1.txt", delimiter=' ')
B = np.genfromtxt("/Users/layaparkavousi/Desktop/B1.txt", delimiter=' ')
C = np.genfromtxt("/Users/layaparkavousi/Desktop/C1.txt", delimiter=' ')

B = B.reshape(N,N,N)
C = C.reshape(N,N,N,N)

A_list = [np.where(A[i,:])[0] for i in range(N)]
B_matrix = [[np.where(B[i,j,:])[0] for j in range(N) ]  for i in range(N)]
C_tensor = [[[np.where(C[k,j,i,:])[0] for i in range(N) ]  for j in range(N)] for k in range(N)]



def f():
    for n in range(N):
        #coupling_sum2 = 0
        #coupling_sum3 = 0
        coupling_sum1 = sum(symengine.sin(y(m)-y(n)) for m in A_list[n]) 
        
        #for l in range(N):
        coupling_sum2 = sum(symengine.sin(2*y(l)-y(m)-y(n)) for l in range(N) for m in B_matrix[n][l]) 
            
            #for q in range(N):
        coupling_sum3 = sum(symengine.sin(y(q)+y(l)-y(m)-y(n)) for q in range(N) for l in range(N) for m in C_tensor[n][l][q]) 
        coupling_term =  (couple_const1) * coupling_sum1 +(K_2) * coupling_sum2 + (K_3)* coupling_sum3 #+
        yield  int_freqs[n] + coupling_term

#initial conditions

"""
g_s = [0.2,0.2,0.2,0.2,0.2,0.2] 
I = jitcsde(f, g_s,n=N)
I.set_initial_value(initial_state,0.0)
I.set_integration_parameters(atol=1e-8,first_step=0.001, max_step=0.01,min_step=1e-13)
"""

#K1 = np.arange(0.08,-0.6,-0.02)
K1 = np.arange(-0.58,-0.1,0.02)
r = np.zeros(len(K1))
i=0
couple_const1 = -0.6
initial_state = 2*np.random.random(N) - 1
I = jitcode(f,n=N)
I.generate_f_C(simplify=False, do_cse=False)#, chunk_size=150) 
I.set_integrator('dopri5')
I.set_initial_value(initial_state,0.0)
data = np.vstack(I.integrate(T) for T in np.arange(0., 100., 0.01))
order_par1 = sum(np.exp(data[-1,:]*1j))/N
r1 = np.absolute(order_par1)
print(r1)

for k in K1:
    couple_const1 = k
    I = jitcode(f,n=N)
    I.generate_f_C(simplify=False, do_cse=False)#, chunk_size=150) 
    I.set_integrator('dopri5')
    initial_state = data[-1,:]
    print(initial_state)
    I.set_initial_value(initial_state,0.0)
    #time = np.arange(0, 10, 0.05)
    data = np.vstack(I.integrate(T) for T in np.arange(0., 100., 0.01))
    order_par = sum(np.exp(data[-1,:]*1j))/N
    r[i] = np.absolute(order_par)
    i=i+1

np.savetxt("/Users/layaparkavousi/Desktop/r(k2=0,increase).txt",r)
np.savetxt("/Users/layaparkavousi/Desktop/k(k2=0,increase).txt",K1)

print(r)
"""z = 1/N*np.sum(np.exp(0+1j*data[-1,:]))
r1 = np.absolute(z)

print(r)"""


"""plt.figure()
for i in range(N):
    plt.plot(np.sin(data[:,i]))
plt.show()
"""
