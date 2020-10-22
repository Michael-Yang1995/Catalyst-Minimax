function output = eg_sc(param, z, tau, y, X, lambda, a)
    ell = param.stepsize;
    theta = param.theta0;
    p = param.p0;
    tol = param.tol;
    diff = 10;
    [n,~] = size(y);
    
    gradient_count = 0;
    
    tol = max(10^(-15), tol);%
    
    T = 0; 
    while diff>tol 
        T = T+1;
        p_old = p;
        theta_old = theta;
        
        %compute new p and theta
        [grad_p, grad_theta] = grad(theta, p, y, X, lambda);
        theta_mid = theta - ell*grad_theta;
        %p_mid = reshape(proj(p + tau*grad_p, a),n,1);
        grad_p = grad_p -tau*(p-z);
        p_mid = reshape(proj_ball(p + ell*grad_p, a),n,1);
        [grad_p, grad_theta] = grad(theta_mid, p_mid, y, X, lambda);
        theta = theta - ell*grad_theta;
        %p = reshape(proj(p + tau*grad_p, a),n,1);
        grad_p = grad_p -tau*(p-z);
        p = reshape(proj_ball(p + ell*grad_p, a),n,1);
        
       
        diff = norm(p_mid-p_old, 2)^2 + norm(theta_mid- theta_old,2)^2;
        gradient_count = gradient_count+4;
        
        
        if T>600
            fprintf('subproblem does not converge\n');
            break;
        end
    end 
    
    %gradient_count
    
    fprintf('subproblem converges at %d iterations\n', T);
    output.p = p_mid;
    output.theta = theta_mid;
    output.count = gradient_count ;
    
    
    
    