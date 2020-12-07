function plotline(x,y,a,b,r)
    x_axis = min(x):0.1:max(x)+0.1;
    hold on
    plot(x, y, 'bx')
    plot(x_axis, a*x_axis + b,'g-')
    plot(x_axis, a*x_axis + b + r,'r-')
    plot(x_axis, a*x_axis + b - r,'r-')
    axis tight equal
    hold off
end