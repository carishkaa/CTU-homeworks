function d = dist(x, a)
% function d = dist(x, a)
%
% INPUT: 
% x: 3-by-1 vector
%    c = x(1:2) center of circle,  
%    r = x(3)   radius of circle 
% a: 2-by-N matrix 
%    N points in R^2 
%
% OUTPUT: 
% d: N-by-1 vector of *signed* distances of all points 
%    from the circumference. 

% discard the code from here and implement the functionality: 
N = size(a, 2); 
d = rand(N, 1); 
 

