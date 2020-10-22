function [grad_x, grad_y] = grad(x, y, sigma, beta, lambda)
% y, X are data; lambda is a parameter; theta, p is current point
    [n,~] = size(x);
    %[m,~] = size(y);
    
    grad_x = -beta./(sigma + y + beta.*x) + lambda*x;
    grad_y = beta.*x./((sigma + y).^2 + beta.*x.*(sigma+y));% - 1*y/norm(y,2);
    
   
    
end
    
