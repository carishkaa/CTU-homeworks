function [x_new] = make_GN_iter(x, a)
% function [x_new] = make_GN_iter(x, a)
%
% makes the Gauss-Newton iteration. 
%
% INPUT:
% x, a   are as usual (see dist.m for explanation) 
%
% x_new is the updated x. 
   
g = dist(x, a);
jacobian = compute_jacobian(x, a);
x_new = x - inv(jacobian'*jacobian)*jacobian'*g;

% df_old = 2*jacobian'* g
% f_new = sum(dist(x_new, a).^2)
% diff = abs((x - x_new))