import numpy as np
import tensorflow as tensor
from tqdm import tqdm
import time
import matplotlib as plt
#SOR METHOD 정의
def SOR_method(u, h, f,omega=1.9):
    u_new = np.copy(u)
    for i in range(1, len(u[0])-1):
        for j in range(1, len(u[0])-1):
            u_new[i, j] = (1 - omega) * u[i, j] + omega * (u_new[i+1, j] + u_new[i-1, j] + u_new[i, j+1] + u_new[i, j-1] - h**2 * f[i-1, j-1]) / 4
    return u_new

# Space 정의
H = 1
dx = H/128 # 그리드 정의
dy = H/128 
nx = int(H/dx)+1 #x-그리드 개수 정의

print(nx)

# Time
dt = 0.1 # Time step

# Physical parameters
nu = 1/100 # viscosity
beta = nu*dt/(2*dx**2)

# Initial conditions
u_lid = 1
u_field = np.zeros((nx+1,nx))
v_field = np.zeros((nx,nx+1))

u_field[-1, :] = 2*u_lid - u_field[-2, :] #usig ghost cell, dirichlet conditions u_lid=1
u_field[0, :] = -u_field[1,:] #using ghost cell u_bottom = 0


u_field_old = np.zeros((nx+1,nx))
v_field_old = np.zeros((nx,nx+1))

u_field_list=[u_field_old,u_field]
v_field_list=[v_field_old,v_field]

reynolds_number = u_lid * H / nu
print(f"Reynolds number: {reynolds_number}")

# x방향 행렬 계산 1,-2,1은 3점 stencil 계수!
Lx = np.zeros((nx-2, nx-2))
Ix = np.eye(nx-2)

for i in range(nx-2):
    Lx[i, i] = -2

    if i == 0:
        Lx[i, i+1] = 1
    elif i == nx-3:
        Lx[i, i-1] = 1
    else:
        Lx[i, i-1] = 1
        Lx[i, i+1] = 1
Ax = (Ix - beta * Lx)

# y방향 행렬 계산 1,-2,1은 3점 stencil 계수!

Ly = np.zeros((nx-1, nx-1))
Iy = np.eye(nx-1)

for i in range(nx-1):
    Ly[i, i] = -2

    if i == 0:
        Ly[i, i+1] = 1
    elif i == nx-2:
        Ly[i, i-1] = 1
    else:
        Ly[i, i-1] = 1
        Ly[i, i+1] = 1

Ay = (Iy - beta * Ly)


t = [-dt,0]

for i in tqdm(range(1, 100)):
        t.append(t[-1] + dt) 
        v_bar = (v_field_list[-1][:-1,:-1]+v_field_list[-1][1:,:-1]+v_field_list[-1][:-1,1:]+v_field_list[-1][1:,1:])/4
        v_bar_old = (v_field_list[-2][:-1,:-1]+v_field_list[-2][1:,:-1]+v_field_list[-2][:-1,1:]+v_field_list[-2][1:,1:])/4

        # u* 구하기 (행렬로 나타냄)
        S = -3/2 * (u_field_list[-1][1:-1, 1:-1] * (u_field_list[-1][1:-1, 2:] - u_field_list[-1][1:-1, :-2]) / (2*dy) + 
                        v_bar[:, 1:-1] * (u_field_list[-1][2:,1:-1] - u_field_list[-1][:-2,1:-1]) / (2*dx)) \
                + 1/2 * (u_field_list[-2][1:-1, 1:-1]*(u_field_list[-2][1:-1, 2:] - u_field_list[-2][1:-1, :-2]) / (2*dy) + 
                        v_bar_old[:, 1:-1] * (u_field_list[-2][2:,1:-1] - u_field_list[-2][:-2,1:-1]) / (2*dx))

        # Ix+beta*Lx (점성항 라플라시안 계산, x방향)
        Ix_Lx = ((Ix  + beta * Lx) @ u_field_list[i][1:-1, 1:-1].T).T
        # boundary condition 보정 (맨 끝은 0, 맨 위는 1)
        Ix_Lx[:,0] = Ix_Lx[:,0] + beta * (u_field_list[i][1:-1, 0])
        Ix_Lx[:,-1] = Ix_Lx[:,-1] + beta * (u_field_list[i][1:-1, -1])

        # boundary condition 보정 R(우변항)에 대해서 (양,옆은 다 0)
        R = dt*S + (Iy + beta * Ly) @ Ix_Lx
        R[0,:] = R[0,:] + beta * (beta * u_field_list[i][0, :-2] + (1-2*beta) * u_field_list[i][0, 1:-1] + beta * u_field_list[i][0, 2:])
        R[-1,:] = R[-1,:] + beta * (beta * u_field_list[i][-1, :-2] + (1-2*beta) * u_field_list[i][-1, 1:-1] + beta * u_field_list[i][-1, 2:]) 

        # boundary condition 보정 좌변항에 있는 항들을 우변항으로 옮김
        R[0,:] = R[0,:] + beta * (-beta * u_field_list[i][0, :-2] + (1+2*beta) * u_field_list[i][0, 1:-1] - beta * u_field_list[i][0, 2:]) # u_n+1일 때 값을 넣어야 하는데 어차피 넣는 것들이 경계 조건을 넣는거라 그냥 u의 값을 사용.
        R[-1,:] = R[-1,:] + beta * (-beta * u_field_list[i][-1, :-2] + (1+2*beta) * u_field_list[i][-1, 1:-1] - beta * u_field_list[i][-1, 2:])
        
        # y방향 선형 solve -> Ay @ psi = R, (Iy-bLy)@psi = (Iy+bLy)@u^n, psi = u*
        psi = np.linalg.solve(Ay, R)
        
        # boundary condition 보정
        psi[:,0] = psi[:,0] + beta * (u_field_list[i][1:-1, 0])
        psi[:,-1] = psi[:,-1] + beta * (u_field_list[i][1:-1, -1])

        # x방향 선형 solve -> Ax @ psi = R, (Ix-bLx)@psi = (Ix+bLx)@u^n, psi = u*
        u_trash = np.linalg.solve((Ix - beta*Lx).T, psi.T).T

        # 계산한 값들 u grid에 부여
        u_sta = np.zeros((nx+1, nx))
        u_sta[1:-1, 1:-1] = u_trash

        # boundary conditions
        u_sta[0,:] = -u_sta[1,]  # bottom wall
        u_sta[-1,:] = 2-u_sta[-2,:]  # top wall


        # v* 구하기
        u_bar = (u_field_list[-1][:-1,:-1]+u_field_list[-1][1:,:-1]+u_field_list[-1][:-1,1:]+u_field_list[-1][1:,1:])/4
        u_bar_old = (u_field_list[-2][:-1,:-1]+u_field_list[-2][1:,:-1]+u_field_list[-2][:-1,1:]+u_field_list[-2][1:,1:])/4

        S = -3/2 * (u_bar[1:-1, :] * (v_field_list[-1][1:-1, 2:] - v_field_list[-1][1:-1, :-2]) / (2*dx) + 
                v_field_list[-1][1:-1, 1:-1] * (v_field_list[-1][2:,1:-1] - v_field_list[-1][:-2,1:-1]) / (2*dx)) \
        + 1/2 * (u_bar_old[1:-1, :]*(v_field_list[-2][1:-1, 2:] - v_field_list[-2][1:-1, :-2]) / (2*dx) + 
                v_field_list[-2][1:-1, 1:-1] * (v_field_list[-2][2:,1:-1] - v_field_list[-2][:-2,1:-1]) / (2*dx))

        Iy_Ly = ((Iy  + beta * Ly) @ v_field_list[-1][1:-1, 1:-1].T).T
        Iy_Ly[:,0] = Iy_Ly[:,0] + beta * (v_field_list[-1][1:-1, 0])
        Iy_Ly[:,-1] = Iy_Ly[:,-1] + beta * (v_field_list[-1][1:-1, -1])
        R = dt*S + (Ix + beta * Lx) @ Iy_Ly
        R[0,:] = R[0,:] + beta * (beta * v_field_list[-1][0, :-2] + (1-2*beta) * v_field_list[-1][0, 1:-1] + beta * v_field_list[-1][0, 2:])
        R[-1,:] = R[-1,:] + beta * (beta * v_field_list[-1][-1, :-2] + (1-2*beta) * v_field_list[-1][-1, 1:-1] + beta * v_field_list[-1][-1, 2:]) 

        R[0,:] = R[0,:] + beta * (-beta * v_field_list[-1][0, :-2] + (1+2*beta) * v_field_list[-1][0, 1:-1] - beta * v_field_list[-1][0, 2:]) # u_n+1일 때 값을 넣어야 하는데 어차피 넣는 것들이 경계 조건을 넣는거라 그냥 u의 값을 사용.
        R[-1,:] = R[-1,:] + beta * (-beta * v_field_list[-1][-1, :-2] + (1+2*beta) * v_field_list[-1][-1, 1:-1] - beta * v_field_list[-1][-1, 2:])

        psi = np.linalg.solve(Ax, R)
        psi[:,0] = psi[:,0] + beta * (v_field_list[-1][1:-1, 0])
        psi[:,-1] = psi[:,-1] + beta * (v_field_list[-1][1:-1, -1])

        v_trash = psi @ np.linalg.inv((Iy - beta * Ly).T)

        v_sta = np.zeros((nx, nx+1))
        v_sta[1:-1, 1:-1] = v_trash

        # boundary conditions
        v_sta[:,0] = -v_sta[:, 1]  # bottom wall
        v_sta[:,-1] = -v_sta[:,-2]  # top wall

        # du*/dx 중앙 차분법 안씀
        u_sta_dx = (u_sta[1:-1, 1:] - u_sta[1:-1, :-1]) / (dx)
        v_sta_dy = (v_sta[1:, 1:-1] - v_sta[:-1, 1:-1]) / (dy)

        del_u = u_sta_dx + v_sta_dy

        pi_old = np.zeros((nx+1, nx+1))

        pi_new = SOR_method(pi_old, dx, del_u)
        iter = 0 
        while np.linalg.norm(pi_old - pi_new) > 1e-5 :
                iter += 1
                pi_old = pi_new.copy()  # 이전 pi 저장
                pi_new = SOR_method(pi_old, dx, del_u)  # 새로운 pi 계산
                pi_new[:,0] = pi_new[:,1] # neumann condition 만족. 안쪽과 바깥쪽이 똑같아서 법선방향 0
                pi_new[:,-1] = pi_new[:,-2]
                pi_new[0,:] = pi_new[1,:]
                pi_new[-1,:] = pi_new[-2,:]

        # u_n+1 찾기
        u_field_new = np.zeros((nx+1, nx))
        u_gar = u_sta[1:-1,1:-1] - (pi_new[1:-1,2:-1]-pi_new[1:-1,1:-2])/dx
        u_field_new[1:-1,1:-1] = u_gar
        u_field_new[0,:] = -u_field_new[1,]  # bottom wall
        u_field_new[-1,:] = 2-u_field_new[-2,:]  # top wall
        u_field_list.append(u_field_new)

        # v_n+1 찾기
        v_field_new = np.zeros((nx, nx+1))
        v_gar = v_sta[1:-1,1:-1] - (pi_new[2:-1,1:-1]-pi_new[1:-2,1:-1])/dy
        v_field_new[1:-1,1:-1] = v_gar
        v_field_new[:,0] = -v_field_new[:,1]  # left wall
        v_field_new[:,-1] = -v_field_new[:,-2]  # right wall
        v_field_list.append(v_field_new)

        u_field_mean = (u_field_new[1:,:]+u_field_new[:-1,:])/2
        v_field_mean = (v_field_new[:,1:]+v_field_new[:, :-1])/2

        base = r'C:\Users\난무적호\Desktop\CSEURP\final'
        np.save(fr'{base}/u_2\u{t[-1]:.2f}', u_field_mean)
        np.save(fr'{base}/v_2\v{t[-1]:.2f}', v_field_mean)
        time.sleep(0.01)  # Simulate some processing time
        