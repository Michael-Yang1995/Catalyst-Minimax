function output = diag_sub(param, w, tau, sigma, beta, lambda, a)
    ell = param.stepsize;
    x = param.x0;
    y = param.y0;
    M = param.M;
    tol = max(10^(-23), param.tol);  tol2 = tol^2*M; 
    [n,~] = size(y);
    
    
    gradient_count = 0;
    diff1 = 10;
    
    %for t=1:ceil(log(1/tol)/log(2)) 
    while diff1 > tol
         x_old1 = x;
         y_old1 = y;
        
         diff = 10;
         x = param.x0;
         z = x;
         T = 0;
         while diff>tol2
             T = T+1;
             alpha = 2/(T+1);
             eta = T*ell/2;
             x_old = x;
             w1 = (1-alpha)*x + alpha*z;
             [grad_x, ~] = grad(w1, y, sigma, beta, lambda);
             x = w1 - ell * grad_x;
             z = z - eta * grad_x;
             diff = norm(x - x_old, 2)^2;
             
             
             %theta_old = theta;
             %[~, grad_theta] = grad(theta, p, y, X, lambda);
             %theta = theta - ell * grad_theta;
             %diff = norm(theta - theta_old, 2)^2;
             
             
             if T>10000
                fprintf('subproblem does not converge\n');
                break;
             end
         end 
         
         T
         gradient_count = gradient_count+T;
         
         %theta
         [~, grad_y] = grad(x, w, sigma, beta, lambda);
         y = reshape(proj(w + tau*grad_y, a),n,1);
         gradient_count = gradient_count+1;
         %grap_p(2,1)
         diff1 = norm(y-y_old1, 2)^2 + norm(x- x_old1,2)^2;
         
    end
    
    output.y = y;
    output.x = x;
    output.count = gradient_count;
    
    