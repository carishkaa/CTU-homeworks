function x = wages_fit_model(t, M)
% function x = wages_fit_model(t, M)
%
% INPUT: N data points specified by
% t : N-by-1 vector, years
% M : N-by-1 vector, wages
%
% OUTPUT:
% x: 2-by-1 vector specifying the estimated model 
% M(t) = x[1] + x[2]*t

A = [ones(size(t)) t];
b = M;
x = A\b;