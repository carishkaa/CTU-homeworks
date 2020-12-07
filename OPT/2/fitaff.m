function [U,C,b0] = fitaff(A,k)
b0 = mean(A, 2);
A = A - b0;
[U, C] = fitlin(A,k);