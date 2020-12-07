function [x_new] = make_GN_iter(x, a)
% function [x_new] = make_GN_iter(x, a)
%
% makes the Gauss-Newton iteration. 
%
% INPUT:
% x, a   are as usual (see dist.m for explanation) 
%
% x_new is the updated x. 
   
 % discard the code from here and implement the functionality: 
x(1:2) = x(1:2) + .1*randn(2, 1); 
x_new = x; 

    

