function [J] = compute_jacobian(x, a)
% function [jacobian] = calculate_jacobian(x, a)
%
% computes the jacobian matrix. 
%
% INPUT:
% x, a   are as usual (see dist.m for explanation) 
%
% OUTPUT:
% jacobian: N-by-3 matrix of the partial derivatives

N = size(a, 2);
d1 = x(1)-a(1,:);
d2 = x(2)-a(2,:);
s = sqrt(d1.^2 + d2.^2);
J = [d1./s; d2./s; -ones(1,N)]';