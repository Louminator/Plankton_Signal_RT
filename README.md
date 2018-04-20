# Plankton_Signal_RT

## 2D code for diffusion simulation using Crank-Nicolson Method
Currently we have diffusion and agents moving with periodic BC.

$\frac{\partial u}{\partial t} = \kappa \nabla^2 u - \beta u + a1\sum_{i=1}^{N}\delta(\mathbf{x}-\mathbf{x_i})$

with periodic boundary conditions.

The above equation's 1D version using the CN can be written as a finite difference equation:
$$
-\alpha u^{n+1}_{i-1} + (1+2\alpha+\beta\Delta t/2)u^{n+1}_{i} - \alpha u^{n+1}_{i-1}
=
\alpha u^{n}_{i-1} + (1-2\alpha-\beta\Delta t/2)u^{n}_{i} + \alpha u^{n}_{i-1}
$$

<span style="color:blue">
There are some little typos in your explanation.  I am going to rework it here in blue. You have the sign flipped in the decay term.  In 2D, there is a 4 on the diagonal because it is $\nabla^2c = c_{xx} + c_{yy}$.
I prefer defining
$$
\alpha = \frac{\kappa \Delta t}{(\Delta x)^2},
$$
rather than
$$
\alpha = \frac{\kappa \Delta t}{2 (\Delta x)^2},
$$
For the rest of this, I will use my definition, but in your parts of the text, I'll leave it as you had it.
To go to 2D, we have
$$
(1+4\alpha/2+\beta\Delta t/2)u^{n+1}_{i,j} - \alpha/2 
\left(u^{n+1}_{i+1,j}+u^{n+1}_{i-1,j}+u^{n+1}_{i,j+1}+u^{n+1}_{i,j-1} \right)
= \\
(1+4\alpha/2-\beta\Delta t/2)u^{n}_{i,j} + \alpha/2
\left( u^{n}_{i+1,j}+u^{n}_{i-1,j}+u^{n}_{i,j+1}+u^{n}_{i,j-1}  \right)
$$
</span>

<span style="color:blue">
The Crank-Nicolson Matrix we will be using is as follows
$$M_{1} = 
\begin{bmatrix}
A_1 & -(\alpha/2) I & 0 & 0 & -(\alpha/2) I \\
-(\alpha/2) I & A_1 & -(\alpha/2) I & 0 & 0 \\
0 & -(\alpha/2) I & A_1 & -(\alpha/2) I & 0 \\
0 & 0 & -(\alpha/2) I & A_1 & -(\alpha/2) I \\
-(\alpha/2) I & 0 & 0 & -(\alpha/2) I & A_1
\end{bmatrix}$$
and
$$M_{2} = 
\begin{bmatrix}
A_2 & (\alpha/2) I & 0 & 0 & (\alpha/2) I \\
(\alpha/2) I & A_2 & (\alpha/2) I & 0 & 0 \\
0 & (\alpha/2)  I & A_2 & (\alpha/2) I & 0 \\
0 & 0 & (\alpha/2) I & A_2 & (\alpha/2) I \\
(\alpha/2) I & 0 & 0 & (\alpha/2) I & A_2
\end{bmatrix}$$
where the matrices $A_1$, $A_2$ are the 1D Crank-Nicolson matrices with periodic boundary condition
$$
A_1 = \begin{bmatrix}
1+2\alpha+\beta\Delta t/2 & -\alpha/2 & 0 & 0 & 0 & -\alpha/2\\
-\alpha/2 & 1+2\alpha+\beta\Delta t/2 & -\alpha/2 & 0 & 0 & 0\\
0 & -\alpha/2 & 1+2\alpha+\beta\Delta t/2 & -\alpha/2 & 0 & 0\\
0 & 0 & -\alpha/2 & 1+2\alpha+\beta\Delta t/2 & -\alpha/2 & 0\\
0 & 0 & 0 & -\alpha/2 & 1+2\alpha+\beta\Delta t/2 & -\alpha/2\\
-\alpha/2 & 0 & 0 & 0 & -\alpha/2 & 1+2(\alpha/2)+\beta\Delta t/2
\end{bmatrix}
$$
$$
A_2 = \begin{bmatrix}
1-2\alpha-\beta\Delta t/2 & \alpha/2 & 0 & 0 & 0 & \alpha/2\\
\alpha/2 & 1-2\alpha-\beta\Delta t/2 & \alpha/2 & 0 & 0 & 0\\
0 & \alpha/2 & 1-2\alpha-\beta\Delta t/2 & \alpha/2 & 0 & 0\\
0 & 0 & \alpha/2 & 1-2\alpha-\beta\Delta t/2 & \alpha/2 & 0\\
0 & 0 & 0 & \alpha/2 & 1-2\alpha-\beta\Delta t/2 & \alpha/2\\
\alpha/2 & 0 & 0 & 0 & \alpha/2 & 1-2\alpha-\beta\Delta t/2
\end{bmatrix}
$$
</span>

The linear equation system we are trying to solve is
$$
M_1 \cdot [u^{n+1}] = M_2 \cdot [u^{n}] + \sum_{i=0}^N f_i \Delta t
$$
where the direc-delta function is 
$$
\delta(x,y) = \frac{1}{2\pi \sigma^2}\exp -{\frac{(x-x_avg)^2+(y-y_avg)^2}{4\sigma^2}}
$$
