
opts.iterN = 10; % change as you like 
opts.VERBOSE = 1; % enable interactive plots 
opts.mu = 1; % for LM 

% these points are for demonstative purposes only. 
% use your own, try different configurations. 
% load my_points_1.mat 



a = [0.6 0.5  -0.6 -0.5 0
     0.5 -0.5 0.5 -0.5  0];

x0 = [0, -0.5]';

method = 'GN'; 


[x, f_history] = fit_circle(x0, a, method, opts);