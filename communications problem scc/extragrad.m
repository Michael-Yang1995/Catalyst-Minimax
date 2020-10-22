function output = extragrad(param, sigma, beta, lambda, a, opt)
    tau = param.stepsize;
    x = param.x0;
    y = param.y0;
    tol = param.tol;
    maxT = param.maxT;
    diff = 10;
    [n,~] = size(y);
    
    x_opt = opt.x; y_opt = opt.y; opt_value = opt.value;
    
    ave_x = x;
    ave_y = y;
    T = 0;
    
    
    %  measure error
    error_value = zeros(1, maxT); error_dist = zeros(1, maxT); error_gm = zeros(1, maxT);
    error_value(1,1) = abs(loss(x, y, sigma, beta, lambda)-opt_value);
    error_dist(1,1) = norm(x - x_opt,2) + norm(y - y_opt,2);
    %error_dist(1,1) = norm(x - x_opt,2)/norm(x_opt, 2) + norm(y - y_opt,2)/norm(y_opt, 2);
    [test_grad_x1, test_grad_y1] = grad(x, y, sigma, beta, lambda);
    error_gm(1,1)=norm(y - reshape(proj(y + 0.1*test_grad_y1, a),n,1))/0.1 + norm(test_grad_x1,2);
    
    
    
    while diff>tol 
        T = T+1;
        x_old = x;
        y_old = y;
        % compute new p and theta
        [grad_x, grad_y] = grad(x, y, sigma, beta, lambda);
        x_mid = x - tau*grad_x;
        y_mid = reshape(proj(y + tau*grad_y,a),n,1);
        [grad_x, grad_y] = grad(x_mid, y_mid, sigma, beta, lambda);
        x = x - tau*grad_x;
        y = reshape(proj(y + tau*grad_y,a),n,1);
        %norm(x_mid-x_old, 2)^2 + norm(y_mid- y_old,2)^2
        
        % update average of p and theta
        ave_x_old = ave_x;
        ave_y_old = ave_y;
        ave_x = (ave_x_old*T + x_mid)/(T+1);
        ave_y = (ave_y_old*T + y_mid)/(T+1);
        diff = norm(ave_x-ave_x_old, 2)^2 + norm(ave_y- ave_y_old,2)^2;
        
        %measure error
        error_value(1,T+1) = abs(loss(ave_x, ave_y, sigma, beta, lambda)-opt_value);
        %error_dist(1,T+1) = norm(ave_x - x_opt,2)/norm(x_opt, 2) + norm(ave_y - y_opt,2)/norm(y_opt, 2);
        error_dist(1,T+1) = norm(ave_x - x_opt,2) + norm(ave_y - y_opt,2);
        [test_grad_x1, test_grad_y1] = grad(ave_x, ave_y, sigma, beta, lambda);
        error_gm(1,T+1)=norm(ave_y - reshape(proj(ave_y + 0.1*test_grad_y1, a),n,1))/0.1 + norm(test_grad_x1,2);
        error_gm(T+1)
        
        if T>maxT
            fprintf('stop before converge\n');
            break;
        end
            
    end 
    fprintf('converge at %d iterations\n', T);
    fprintf('converge with %d accuracy\n', diff);
    fprintf('converge with %d accuracy\n', error_dist(T+1));
    output.x = ave_x;
    output.y = y;
    output.error_value = error_value;
    output.error_dist = error_dist;
    output.error_gm = error_gm;
        
    
    