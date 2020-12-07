function [x, f_history] = fit_circle(x, a, method, opts)
%
%
if opts.VERBOSE
    clc, close all
    plot(a(1,:),a(2,:),'bx','MarkerSize',10);
    xlim([-1,1])
    ylim([-1,1])
    hold on
    axis equal
    
    new_x = [x; compute_r(x, a)];
    h = plot_circle(new_x);
    
    title('Initialization')
    disp('=== Press any key ===')
    pause
end

f0 = compute_criterion(x, a); % at init
f_history = zeros(1, opts.iterN+1); 
f_history(1) = f0; 


if strcmp(method, 'GN')
else
    mu = opts.mu; 
end

for iter = 1:opts.iterN 
   if strcmp(method, 'GN')
       x = GN_iter(x, a);
   else % LM 
       [x, success] = make_LM_iter(x, a, mu);
       if success
           mu = mu/3; 
       else
            mu = mu*3; 
       end
   end

   f = compute_criterion(x, a); 
   f_history(iter+1) = f; 

   if opts.VERBOSE 
       delete(h) 
       
%        h = plot_circle(x); 
       new_x = [x; compute_r(x, a)];
       h = plot_circle(new_x);
        
       title(sprintf('Iteration %i', iter))
       disp('=== Press any key ===')
       pause
   end

   % is radius >0 ?
%    if x(3) <=0 
%        x(3) = std(a(:)); %reinitialize
%    end
   
end
hold off

end

    
function h = plot_circle(x); 
    [c, r] = deal( x(1:2), x(3));
    phi = linspace(0, 2*pi, 100); 
    h = plot(r*cos(phi)+c(1), r*sin(phi)+c(2), 'r');
end

function r = compute_r(c, a)
    N = size(a, 2);
    d1 = c(1)-a(1,:);
    d2 = c(2)-a(2,:);
    r = sqrt(d1.^2 + d2.^2)' ./ N;
    r = sum(r);
end

function [f] = compute_criterion(x, a)

%[c, r] = deal( [x(1); x(2)], r(x, a));
% residuals 
%g = dist(x, a); 
g = dist(x, a); 
% criterion 
f = sum(g.^2); 
end


function [c_new] = GN_iter(c, a)

g = dist(c,a);
jacobian = compute_jacobian(c, a);
c_new = c - inv(jacobian'*jacobian)*jacobian'*g;

% g = dist(c_new, a); 
% f = sum(g.^2)
% df = 2.*g'* jacobian
end


function [J] = compute_jacobian(x, a)
N = size(a, 2);
d1 = x(1)-a(1,:);
d2 = x(2)-a(2,:);
s = sqrt(d1.^2 + d2.^2);
J = [d1./s - sum(d1./s)/N; d2./s - sum(d2./s)/N]';
end

function d = dist(c, a)
N = size(a, 2);
d1 = c(1)-a(1,:);
d2 = c(2)-a(2,:);
r = sum(sqrt(d1.^2 + d2.^2))/ N;
d = sqrt(d1.^2 + d2.^2)' - r;
end
