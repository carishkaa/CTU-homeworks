function [a,b,r] = minimaxfit(x,y)
    [n, m] = size(x);
    f = [zeros(n,1)' 0 1];
    A = [x', ones(m,1), -ones(m,1); -x', -ones(m,1), -ones(m,1)];
    Y = [y -y];
    result = linprog(f, A, Y);
    a = result(1:n);
    b = result(n+1);
    r = result(n+2);
end