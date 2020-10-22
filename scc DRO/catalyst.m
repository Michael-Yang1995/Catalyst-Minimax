function output = catalyst(param, y, X, lambda, a, opt, y_test, X_test)
    tau = param.stepsize;
    theta = param.theta0;
    p = param.p0;
    tol = param.tol;
    ell = param.stepsize_sub; %stepsize for subproblem
    kappa = param.kappa;      %convergence parameter for subproblem
    maxT = param.maxT;
    p_opt = opt.p; theta_opt = opt.theta; opt_value = opt.value;
    
    
    diff = 10;
    [n,~] = size(y);
    [m,~] = size(theta);
    
    T = 0;
    alpha = 1;
    sum_alpha = 1;
    v = p;
    ave_theta = zeros(m,1);
    
    gradient_count = zeros(1, maxT+1);
    %  measure error
    error_gm = zeros(1, maxT); error_value = zeros(1, maxT); error_dist = zeros(1, maxT);
    error_value(1,1) = abs(loss(theta, p, y, X, lambda)-opt_value);
    error_dist(1,1) = norm(theta - theta_opt,2) + norm(p - p_opt,2);
    %error_dist(1,1) = norm(theta - theta_opt,2)/norm(theta_opt,2) + norm(p - p_opt,2)/norm(p_opt,2);
    [test_grad_p1, test_grad_theta1] = grad(theta, p, y, X, lambda);
    error_gm(1,1)=norm(p - reshape(proj_ball(p + 0.1*test_grad_p1, a),n,1))/0.1 + norm(test_grad_theta1,2);
    
    while diff>tol
        T = T+1;
        z = alpha*v +(1-alpha)*p;
        p_old = p;
        ave_theta_old = ave_theta;
        
        
        param_sub.theta0 = theta;
        param_sub.p0 = z;
        param_sub.tol = kappa*tau/(T^8);  % subproblem accuracy
        param_sub.stepsize = ell;
        update = eg_sc(param_sub, z, tau, y, X, lambda, a);
        update.count
        gradient_count(1,T+1) = gradient_count(T) + update.count;
        p = update.p;
        theta = update.theta;
        
        ave_theta = (ave_theta*sum_alpha + theta/alpha)/(sum_alpha+1/alpha);
        sum_alpha = sum_alpha + 1/alpha;
        diff = norm(p-p_old, 2)^2 + norm(ave_theta- ave_theta_old,2)^2;
        
        v = p_old + 1/alpha*(p-p_old);
        alpha = (-alpha^2 +sqrt(alpha^4+4*alpha^2))/2; 
        
        %measure error
        error_value(1,T+1) = abs(loss(ave_theta, p, y, X, lambda)-opt_value);
        error_dist(1,T+1) = norm(ave_theta - theta_opt,2) + norm(p - p_opt,2);
        %error_dist(1,T+1) = norm(ave_theta - theta_opt,2)/norm(theta_opt,2) + norm(p - p_opt,2)/norm(p_opt,2);
        [test_grad_p1, test_grad_theta1] = grad(ave_theta, p, y, X, lambda);
        error_gm(1,T+1)=norm(test_grad_theta1,2)+ norm(p - reshape(proj_ball(p + 0.1*test_grad_p1, a),n,1))/0.1 ;
        error_gm(T+1)
        
        
        if T>maxT
            fprintf('stop before converge\n');
            break;
        end
        
    end
    fprintf('converge at %d iterations\n', T);
    fprintf('converge with %d accuracy\n', diff);
    output.p = p;
    output.theta = ave_theta;
    output.gradient_count = gradient_count;
    output.error_gm = error_gm;
    output.error_value = error_value;
    output.error_dist = error_dist;
    
    