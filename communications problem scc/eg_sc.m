function output = eg_sc(param, z, tau, sigma, beta, lambda, a)
    ell = param.stepsize;
    x = param.x0;
    y = param.y0;
    tol = param.tol;
    diff = 10;
    [n,~] = size(y);
    
    gradient_count = 0;
    
    tol = max(10^(-8), tol);%
    
    T = 0; 
    while ((diff>tol) || (T<0))
        T = T+1;
        y_old = y;
        x_old = x;
        
        %compute new p and theta
        [grad_x, grad_y] = grad(x, y, sigma, beta, lambda);
        x_mid = x - ell*grad_x;
        grad_y = grad_y -tau*(y-z);
        y_mid = reshape(proj(y + ell*grad_y,a),n,1);
        [grad_x, grad_y] = grad(x_mid, y_mid, sigma, beta, lambda);
        x = x - ell*grad_x;
        grad_y = grad_y - tau*(y-z);
        y = reshape(proj(y + ell*grad_y, a),n,1);
        
       
        diff = norm(y-y_old, 2)^2 + norm(x- x_old,2)^2;
        gradient_count = gradient_count+4;
        
        
        if T>6000
            fprintf('subproblem does not converge\n');
            break;
        end
    end 
    
    %gradient_count
    
    fprintf('subproblem converges at %d iterations\n', T);
    output.x = x;
    output.y = y;
    output.count = gradient_count ;
    
    
    
    