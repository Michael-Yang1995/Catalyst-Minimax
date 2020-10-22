function [grad_p, grad_theta] = partial_grad(theta, p, y, X, lambda, beta)
% y, X are data; lambda is a parameter; theta, p is current point
    [n,~] = size(p);
    [m,~] = size(theta);
    
    v = (X*theta).*y;   %v_i is y_i*theta^T*X_i))
    if beta==0
       grad_p = n*log(1+exp(-v));   % already times n here 
       grad_theta = n*(repmat((-p.*y)./(exp(v)+1),1,m).*X).' + repmat(lambda*theta, 1,n); % already times n here
    else
       grad_p = zeros(n,1); grad_p(beta, 1) = n*log(1+exp(-v(beta)));
       %theta0 = zeros(m,1);  theta0(alpha,1) = theta(alpha,1);
       grad_theta = n*(-p(beta)*y(beta)/(exp(v(beta))+1)*X(beta,:)).' + lambda*theta; 
    end
end