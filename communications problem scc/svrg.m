function output = svrg(param, sigma, beta, lambda, a, opt)
    tau = param.stepsize;
    x = param.x0;
    y = param.y0;
    tol = param.tol;
    maxT = param.maxT;
    M = param.M;
    x_opt = opt.x; y_opt = opt.y; opt_value = opt.value;
    
    
    diff = 10;
    [n,~] = size(y);
    
    T = 0;
    gradient_count = zeros(1, maxT+1);
    
    %  measure error
    error_value = zeros(1, maxT); error_dist = zeros(1, maxT); error_gm =zeros(1, maxT);
    error_value(1,1) = abs(loss(x, y, sigma, beta, lambda)-opt_value);
    error_dist(1,1) = norm(x - x_opt,2) + norm(y - y_opt,2);
    [test_grad_x1, test_grad_y1] = grad(x, y, sigma, beta, lambda);
    error_gm(1,1)=norm(y - reshape(proj(y + 0.1*test_grad_y1, a),n,1))/0.1 + norm(test_grad_x1,2);
    
    while diff>tol
        T = T+1;
        x_old = x;
        y_old = y;
        
        
        % record point and full gradients
        til_x = x; til_y = y;
        [full_grad_x, full_grad_y] = grad(til_x, til_y, sigma, beta, lambda);
        full_grad_x = full_grad_x - lambda*x;
        
        %x_fake = x - 0.3*full_grad_x;
        %y_fake = reshape(proj(y + 0.3*(full_grad_y), a),n,1);
        %diff1 = norm(y-y_fake, 2)^2 + norm(x- x_fake,2)^2
        
        
        
        for m= 1:M
            k = randi(n); %alpha = randi(m);
            [current_grad_x, current_grad_y] = partial_grad(x, y, sigma, beta, k);
            
            u = zeros(n,1); u(k) = full_grad_x(k)*n;
            x = x - tau*(current_grad_x - u + full_grad_x + lambda*x);
            v = zeros(n,1); v(k) = full_grad_y(k)*n;
            y = y + tau*(current_grad_y - v + full_grad_y );
            y = reshape(proj(y, a), n,1);
             
        
        end
        gradient_count(T+1) = gradient_count(T) + 2 + 2*M/n;
        %measure error
        error_value(1,T+1) = abs(loss(x, y, sigma, beta, lambda)-opt_value);
        error_dist(1,T+1) = norm(x - x_opt,2) + norm(y - y_opt,2);
        [test_grad_x1, test_grad_y1] = grad(x, y, sigma, beta, lambda);
        error_gm(1,T+1)=norm(y - reshape(proj(y + 0.1*test_grad_y1, a),n,1))/0.1 + norm(test_grad_x1,2);
        error_gm(1,T+1)
        
        
        if T>maxT
            fprintf('stop before converge\n');
            break;
        end
        
        diff = norm(y-y_old, 2)^2 + norm(x- x_old,2)^2;
        
    end
    fprintf('converge at %d iterations\n', T);
    fprintf('converge with %d accuracy\n', diff);
    output.y = y;
    output.x = x;
    output.gradient_count = gradient_count;
    output.error_value = error_value;
    output.error_dist = error_dist;
    output.error_gm = error_gm;

    
    