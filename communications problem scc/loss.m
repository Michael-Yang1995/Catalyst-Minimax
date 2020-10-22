function a = loss(x, y, sigma, beta, lambda)
% y, X are data; lambda is a parameter; theta, p is current point
    a = -sum(log(1+beta.*x./(sigma+y))) + lambda*norm(x,2)^2/2;
end
    