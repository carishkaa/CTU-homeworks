function drawfitline(A)
n = size(A, 2);
[U, C, b0] = fitaff(A, 1);
B = U*C + b0;

ax = A(1,:); 
ay = A(2,:);
bx = B(1,:);
by = B(2,:);
x = 1:10;

hold on
plot(ax, ay, 'rx')
plot(x,(U(2)/U(1))*(x - b0(1))+b0(2),'g')
for i = 1:n
    plot([ax(i) bx(i)], [ay(i) by(i)],'r-')
end
axis equal

return