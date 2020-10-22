function output = diag_sub(param, w, tau, y, X, lambda, a)
    ell = param.stepsize;
    p = param.p0;
    theta = param.theta0;
    M = param.M;
    tol = param.tol;  tol2 = tol^2*M; 
    [n,~] = size(y);
    
    gradient_count = 0;
    diff1 = 10;
    
    %for t=1:ceil(log(1/tol)/log(2)) 
    while diff1 > tol
         theta_old1 = theta;
         p_old1 = p;
        
         diff = 10;
         theta = param.theta0;
         z = theta;
         T = 0;
         while diff>tol2
             T = T+1;
             alpha = 2/(T+1);
             eta = T*ell/2;
             theta_old = theta;
             w1 = (1-alpha)*theta + alpha*z;
             [~, grad_theta] = grad(w1, p, y, X, lambda);
             theta = w1 - ell * grad_theta;
             z = z - eta * grad_theta;
             diff = norm(theta - theta_old, 2)^2;
             
             
             %theta_old = theta;
             %[~, grad_theta] = grad(theta, p, y, X, lambda);
             %theta = theta - ell * grad_theta;
             %diff = norm(theta - theta_old, 2)^2;
             
             
             if T>1000
                fprintf('subproblem does not converge\n');
                break;
             end
         end 
         
         T
         gradient_count = gradient_count+T;
         
         %theta
         [grad_p, ~] = grad(theta, w, y, X, lambda);
         p = reshape(proj_ball(w + tau*grad_p, a),n,1);
         gradient_count = gradient_count+1;
         %grap_p(2,1)
         diff1 = norm(p-p_old1, 2)^2 + norm(theta- theta_old1,2)^2;
         
    end
    
    output.p = p;
    output.theta = theta;
    output.count = gradient_count;
    