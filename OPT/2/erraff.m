function d = erraff(A)
[m, n] = size(A);
b0 = mean(A, 2);
A = A - b0*ones(1,n);
[~, D] = eig(A * A');
d = zeros(1,m);
for i = 1:m
    d(i) = trace(D(1:m-i,1:m-i));
end
