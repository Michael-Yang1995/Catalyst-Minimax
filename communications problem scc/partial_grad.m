function [grad_x, grad_y] = partial_grad(x, y, sigma, beta, k)
% y, X are data; lambda is a parameter; theta, p is current point
    [n,~] = size(x);
    %[m,~] = size(y);
    
    grad1_x = n*(-beta(k)/(sigma(k) + y(k) + beta(k)*x(k)));
    grad1_y = n*beta(k)*x(k)/((sigma(k) + y(k))^2 + beta(k)*x(k)*(sigma(k)+y(k)));
    
    grad_x = zeros(n,1); 
    grad_x(k) = grad1_x;
    grad_y = zeros(n,1);
    grad_y(k) = grad1_y;
end