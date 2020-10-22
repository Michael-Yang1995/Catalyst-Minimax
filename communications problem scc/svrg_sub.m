function output = svrg_sub(param, z, tau, sigma, beta, lambda, a)
    ell = param.stepsize;
    x = param.x0;
    y = param.y0;
    tol = param.tol;
    M = param.M;
    diff = 10;
    [n,~] = size(y);
    
    tol = max(10^(-11.5), tol);
    
    gradient_count = 0;
    
    T = 0; 
    while diff>tol 
        T = T+1;
        x_old = x;
        y_old = y;
        
        % record point and full gradients
        til_x = x; til_y = y;
        [full_grad_x, full_grad_y] = grad(til_x, til_y, sigma, beta, lambda);
        full_grad_x = full_grad_x - lambda*x;
        
        
        
        %check whether need to stop
        x_fake = x - 0.3*full_grad_x;
        y_fake = reshape(proj(y + 0.3*(full_grad_y- tau*(y-z)), a),n,1);
        diff1 = norm(y-y_fake, 2)^2 + norm(x- x_fake,2)^2;
        
        if diff<tol
            break;
        end
        
        gradient_count = gradient_count + 2;
        
        
        for m= 1:M
            k = randi(n);
            [current_grad_x, current_grad_y] = partial_grad(x, y, sigma, beta, k);
                        
            u = zeros(n,1); u(k) = full_grad_x(k)*n;
            x = x - ell*(current_grad_x - u + full_grad_x + lambda*x);
            v = zeros(n,1); v(k) = full_grad_y(k)*n;
            y = y + ell*(current_grad_y - v + full_grad_y - tau*(y-z));
            y = reshape(proj(y, a), n,1);    
            %check whether need to stop
            %theta_fake = theta - 0.3*full_grad_theta;
            %p_fake = reshape(proj_ball(p + 0.3*(full_grad_p- tau*(p-z)), a),n,1);
            %diff = norm(p-p_fake, 2)^2 + norm(theta- theta_fake,2)^2
            %if diff<tol
            %   break;
            %end
            
        
        end
        
        diff = norm(y-y_old, 2)^2 + norm(x- x_old,2)^2;
        gradient_count = gradient_count + 2*M/n;  
        
        if T>500
            fprintf('subproblem does not converge\n');
            break;
        end
        
    end 
    
    
    %gradient_count
    
    fprintf('subproblem converges at %d iterations\n', T);
    output.y = y;
    output.x = x;
    output.count = gradient_count;
    
    
    
    