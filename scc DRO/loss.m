function a = loss(theta, p, y, X, lambda)
% y, X are data; lambda is a parameter; theta, p is current point
    [n,~] = size(p);
    [m,~] = size(theta);
    v = (X*theta).*y;  
    a = p.'*log(1+exp(-v)) + lambda/2*norm(theta,2)^2;
end
    