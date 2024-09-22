# -*- coding: utf-8 -*-
import numpy as np
import sympy as sp
import math
import random

def orthoset(arry):
    oset = 0
    for i in range(2):
        for j in range(1,3):
            if i!=j:
                if np.dot(arry[i],arry[j]) == 0:
                    print(f'{arry[i]} and {arry[j]} form Orthogonal Sets ')
                    oset+=1
    if oset == 0:
        print("There are no orthogonal sets in the matrix")
        return 0

def inv(arr):
    if round(np.linalg.det(arr)) !=0:
        print("The inverse of matrix\n",np.linalg.inv(arr))
    else:
        print("As determinant of matrix = 0,hence the inverse of matrix does not exist\n")  
    return 0

def linearCom (arr1, arr2):
    for j in range(3):
        i=0
        l1 = arr1[i][j]/arr2[i][i]
        l2 = arr1[i+1][j]/arr2[i+1][i]
        l3 = arr1[i+2][j]/arr2[i+2][i]
        print(f'\nThe linear combination of {j+1}th  coloumn of martix of A with d are x = {l1}, y = {l2} and z = {l3}')
    return 0
#Question A.
print("Sahil Sheikh")
print("CWID:   A20518693")
print("Subject:CS 577")
print("Semester: FALL 22")
print("\n\nASSIGNMENT 0")
a = np.array([[1],[2],[3]])
b = np.array([[4],[5],[6]])
c = np.array([[-1],[1],[3]])
x = np.array([1,1,1])
seq = [*range(-100,100)]
seq.remove(0)
 
# A 1.
print("\nA 1.\n2*a-b = \n",2*a-b)
unit_vec = a/ np.linalg.norm(a)
# A 2.
print("\nA 2.\nThe unit vetor in the direction of a:\n",a/ np.linalg.norm(a))
#A 3.
xaxis = np.array([[1],[0],[0]])
n = np.linalg.norm(a)
theta =math.acos((np.dot(a.T,xaxis))/n*np.linalg.norm(xaxis))
print(f'\nA 3.\nThe norm of a (||a||) = {n}')
theta_deg = math.degrees(theta)
print(f'The angle between a & X-axis in radians ={theta} and in degree ={theta_deg} ')
#A 4.
print("\nA 4.\nthe direction cosines of a:\n",a/n)
#A 5.
theta_ab = np.arccos(np.dot(a.flatten(),b)/(n*np.linalg.norm(b)))
print("\nA 5.\nThe angle between a&b =",math.degrees(theta_ab))
#A 6.
a_b = np.dot(a.flatten(),b)
b_a = np.dot(b.flatten(),a)
print("\nA 6.\na.b : ",a_b)
print("b.a : ",b_a)
#A 7.
print("\nA 7.\na.b using the angle between a&b =",math.cos(theta_ab)*n*np.linalg.norm(b))
#A 8.
print("\nA 8.\nScalar projection of b onto a =",np.dot(unit_vec.T, b)/np.linalg.norm(unit_vec))
#A 9.
while True:
    
   x[0] = random.choice(seq)
   x[1] = random.choice(seq)
   x[2] = random.choice(seq)
   cond = np.dot(a.T,x)
   xn = np.linalg.norm(x)
   a0 = xn*n
   if cond/a0 == 0:
       break
print("\nA 9.\nThe vector perpandicular to a is:",x)
#A 10.
print("\nA 10.\na X b :",np.cross(a.flatten(),b.flatten()))
print("b X a :",np.cross(b.flatten(),a.flatten()))
#A 11.
print("\nA 11.\nThe vector perpandicular to both a and b will be the resultant vector of axb \n",np.cross(a.T,b.T))
#A 12.
abc = np.hstack((a,b,c)) #Stacking matrix a,b,c into a single matrix abc
det_abc = round(np.linalg.det(abc))
if det_abc == 0:
    print("\nA 12.\na,b,c are lineraly dependent as the determinant of the combined matrix a,b,c =",det_abc)
else:
    print("\nA 12.\na,b,c are linealy independent as the determinant of the combined matrix a,b,c =",det_abc)
#A 13.
print("\nA 13. at.b:\n",np.dot(a.T,b))
print("a.bt:\n",np.dot(a,b.T))


#Question B
A = np.array([[1,2,3],[4,-2,3],[0,5,-1]])
C = np.array([[1,2,3],[4,5,6],[-1,1,3]])
B = np.array([[1,2,1],[2,1,-4],[3,-2,1]])
d = np.array([[1],[2],[3]])

print("\nB 1.\n2A-B:\n",2*A-B)
print("\nB 2.\nThe dot product of AB:\n",np.dot(A,B))
print("The dot product of BA:\n",np.dot(B,A))
print("\nB 3.\nTranspose of AB :\n ",np.transpose(np.dot(A,B)))
print("TransposeB dot product with transposeA: \n",np.dot(np.transpose(B), np.transpose(A)))
print("\nB 4.\n|A| : ",round(np.linalg.det(A)))
print("|C| : ",round(np.linalg.det(C)))
print("\nB 5.\n ")
print("In matrix A")
orthoset(A)
print("In matrix B")
orthoset(B)
print("In matrix C")
orthoset(C)  
print("\nB 6.\nFor matrix A")
inv(A)
print("\nFor matrix B")
inv(B)
print("\nB 7.\nFor matrix C:\n")
inv(C)
print("\nB 8.\nThe dot product of A and d(Ad):\n",np.matmul(A, d))
d_nor = d/np.linalg.norm(d)
print("\nB 9.\nNormalizing d :\n",d_nor)
print("Scalar projection of rows of A onto normalized d:\n",np.dot(A[0], d_nor)/np.linalg.norm(d_nor), np.dot(A[1], d_nor)/np.linalg.norm(d_nor), np.dot(A[2], d_nor)/np.linalg.norm(d_nor))
aAB = np.arccos(np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B)))
print("\nB 10.\nVector projection of rows of A onto normalized d:\n",np.dot(A[0],d_nor)/(np.linalg.norm(d_nor)**2)*d_nor)
print("\n",np.dot(A[1],d_nor)/(np.linalg.norm(d_nor)**2)*d_nor)
print("\n",np.dot(A[2],d_nor)/(np.linalg.norm(d_nor)**2)*d_nor)

print("\nB 11.\n")
linearCom(A, d)
print("\nB 12.\n Value of x for Bx = d :\n",np.around(np.linalg.solve(B,d),1))
print("\nB 13.\nx for Cx = d: Does not exsist as |C| = 0. Therefore inverse of C does not exsist\n")

#Question C

E = np.array([[2,-2],[-2,5]])
F = np.array([[1,2],[2,4]])
D = np.array([[1,2],[3,2]])
Z = np.array([[0],[0]])
I = np.array([[1,0],[0,1]])
x1 = np.array([[0.0],[1.0]])
y1 = np.array([[0.0],[1.0]])
ev,evc = np.linalg.eig(D)
lam1 = ev[0]*I
lam2 = ev[1]*I
d1 = sp.Matrix([[1,2],[3,2]])
d_lam1 = d1 - lam1
d_lam1_rff = d_lam1.rref()
x1[0][0] = ((-1.0)*d_lam1_rff[0][1])/d_lam1_rff[0][1]
print(f'\nC 1.\nEigen vectors of D for lamda = {ev[0]} is [{x1[0][0]} , {x1[1][0]}]')
d_lam2 = d1 - lam2
d_lam2_rff = d_lam2.rref()
y1[0][0] = ((-1.0)*d_lam2_rff[0][1]/d_lam2_rff[0][0])
print(f'Eigen vectors of D for lamda = {ev[1]} is [{y1[0][0]} , {y1[1][0]}]')

print("\nC 2.\nDot product of Eigen vectors of D:",np.dot(x1.T,y1))
x2 = np.array([[0.0],[1.0]])
y2 = np.array([[0.0],[1.0]])
eve,evce = np.linalg.eig(E)
le1 = eve[0]*I
le2 = eve[1]*I
e1 = sp.Matrix([[2,-2],[-2,5]])
el1 = e1 - le1
el1_rff = el1.rref()

el2 = e1 - le2
el2_rff = el2.rref()
y2[0][0] = (-1)*el2_rff[0][1]/el2_rff[0][0]
x2[0][0] = (-1)*el1_rff[0][1]/el1_rff[0][0]
print("\n3.\n")
print(f'Eigen vectors of E for lamda = {eve[0]} is [{x2[0][0]} , {x2[1][0]}]')
print(f'Eigen vectors of D for lamda = {eve[1]} is [{y2[0][0]} , {y2[1][0]}]')
print("\nThe dot product of eigen vectors of E =",np.dot(x2.T,y2))
print("\n4. Since dot product of eigen vectors of E is 0 that implies they are orthogonal and since the dot product is 0 it also means they are perpandicular")
print("\n5.\n Rank of F: ",np.linalg.matrix_rank(F))
f1 = sp.Matrix([[1,2],[2,4]])
f1_rff = f1.rref()
f10 = np.array([[1,2,0],[2,4,0]])
r_f10 = np.linalg.matrix_rank(f10)
r_F = np.linalg.matrix_rank(F)
print("For matrix F in F.x=0")
if (r_F == r_f10):
    if r_F == r_f10 == len(F[0]):
        print(" matrix has unique solution")
    else:
        if r_F == r_f10 < len(F[0]):
            print("Matrix has infinite soln")
else:
    print("Matrix has no solution")    
xf1 = (-1)*f1_rff[0][1]*1
xf2 = (-1)*f1_rff[0][1]*2
print(f'\n6.\nThe non trivial solution of F are {xf1},{1} and {xf2},2')
r_d = np.linalg.matrix_rank(D)
d0 = np.array([[1,2,0],[3,2,0]])
dor = np.linalg.matrix_rank(d0)
if (r_d == dor):
    if r_d == dor == len(D[0]):
        print(f'\n7.\nmatrix D has unique solution as rank of d: {r_d} = rank of D|0:{dor} = no of unknown {len(D[0])} and its trivial solution = 0,0 ')
    else:
        if r_d == dor < len(D[0]):
            print("Matrix has infinite soln")
else:
    print("Matrix has no solution") 






