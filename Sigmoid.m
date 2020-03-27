xi = linspace(-0.1,pi,1000);

c_Ltotal = C_Ltotal(xi);
c_Dairfoil = C_Dairfoil(xi);
c_Dflatplate = C_Dflatplate(xi);
%sigmoid_neg = 1 - sigmoid;

%c_L = c_Lflatplate*sigmoid;
sigi = sigmoid(xi);

plot(xi,c_Ltotal)