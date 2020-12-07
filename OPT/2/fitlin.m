function [U,C] = fitlin(A, k)
m = size(A, 1);
[V, ~] = eig(A * A');
U = V(:, m-k+1:end);
C = U'*A;