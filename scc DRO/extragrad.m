function output = extragrad(param, y, X, lambda, a, opt, y_test, X_test)
    tau = param.stepsize;
    theta = param.theta0;
    p = param.p0;
    tol = param.tol;
    maxT = param.maxT;
    diff = 10;
    [n,~] = size(y);
    [~,m] = size(X);
    
    p_opt = opt.p; theta_opt = opt.theta; opt_value = opt.value;
    
    ave_p = zeros(n,1);
    ave_theta = zeros(m,1);
    T = 0;
    
    
    %  measure error
    error_gm = zeros(1, maxT); error_value = zeros(1, maxT); error_dist = zeros(1, maxT);
    error_value(1,1) = abs(loss(theta, p, y, X, lambda)-opt_value);
    error_dist(1,1) = norm(theta - theta_opt,2) + norm(p - p_opt,2);
    %error_dist(1,1) = norm(theta - theta_opt,2)/norm(theta_opt,2) + norm(p - p_opt,2)/norm(p_opt,2);
    [test_grad_p1, test_grad_theta1] = grad(theta, p, y, X, lambda);
    error_gm(1,1)=norm(p - reshape(proj_ball(p + 0.1*test_grad_p1, a),n,1))/0.1 + norm(test_grad_theta1,2);
    
    
    while diff>tol 
        T = T+1;
        % compute new p and theta
        [grad_p, grad_theta] = grad(theta, p, y, X, lambda);
        theta_mid = theta - tau*grad_theta;
        %p_mid = reshape(proj(p + tau*grad_p, a),n,1);
        p_mid = reshape(proj_ball(p + tau*grad_p, a),n,1);
        [grad_p, grad_theta] = grad(theta_mid, p_mid, y, X, lambda);
        theta = theta - tau*grad_theta;
        %p = reshape(proj(p + tau*grad_p, a),n,1);
        p = reshape(proj_ball(p + tau*grad_p, a),n,1);
        
        % update average of p and theta
        ave_p_old = ave_p;
        ave_theta_old = ave_theta;
        ave_p = (ave_p_old*T + p_mid)/(T+1);
        ave_theta = (ave_theta_old*T + theta_mid)/(T+1);
        diff = norm(ave_p-ave_p_old, 2)^2 + norm(ave_theta- ave_theta_old,2)^2;
        
        %measure error
        error_value(1,T+1) = abs(loss(ave_theta, ave_p, y, X, lambda)-opt_value);
        error_dist(1,T+1) = norm(ave_theta - theta_opt,2) + norm(ave_p - p_opt,2);
        [test_grad_p1, test_grad_theta1] = grad(ave_theta, ave_p, y, X, lambda);
        error_gm(1,T+1)=norm(test_grad_theta1,2)+ norm(ave_p - reshape(proj_ball(ave_p + 0.1*test_grad_p1, a),n,1))/0.1 ;
        error_gm(T+1)
        
        
        if T>maxT
            fprintf('stop before converge\n');
            break;
        end
            
    end 
    fprintf('converge at %d iterations\n', T);
    fprintf('converge with %d accuracy\n', diff);
    output.p = ave_p;
    output.theta = ave_theta;
    output.error_gm = error_gm;
    output.error_value = error_value;
    output.error_dist = error_dist;
        
    
    