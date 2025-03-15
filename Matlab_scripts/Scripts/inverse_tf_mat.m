% Define the transfer functions
A = [1, -0.33]; % A(z^-1) = 1 - 0.33z^-1
B = [0, 0.22];  % B(z^-1) = 0.22z^-1
C = [1, 0.15];  % C(z^-1) = 1 + 0.15z^-1
D = [0.31, 0.23]; % D(z^-1)

% Convert these to transfer functions using tf
A_tf = tf(A, A);
B_tf = tf(B, A);
C_tf = tf(C, A);
D_tf = tf(D, A);

% Create the system matrix
G = [
    [B_tf, C_tf, B_tf];
    [D_tf, B_tf, D_tf];
    [B_tf, D_tf, B_tf]
]

% Compute the inverse of G

f = @() inv(G);
timeit(f)
G_inv = inv(G)
