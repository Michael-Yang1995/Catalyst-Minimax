function [grad_p, grad_theta] = grad(theta, p, y, X, lambda)
% y, X are data; lambda is a parameter; theta, p is current point
    [n,~] = size(p);
    [m,~] = size(theta);
    v = (X*theta).*y;   %v_i is log(1+exp(-y_1*theta^T*X_i))
    grad_p = log(1+exp(-v));
    grad_theta = sum(repmat((-p.*y)./(exp(v)+1),1,m).*X,1).' + lambda*theta;
end
    
