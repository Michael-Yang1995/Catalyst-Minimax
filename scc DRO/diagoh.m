function output = diagoh(param, y, X, lambda, a, opt, y_test, X_test)
    tau = param.stepsize;
    theta = param.theta0;
    p = param.p0;
    tol = param.tol;
    ell = param.stepsize_sub; %stepsize for subproblem
    kappa = param.kappa;      %convergence parameter for subproblem
    M = param.M;         %convergence parameter for subproblem' subproblem
    maxT = param.maxT;
    p_opt = opt.p; theta_opt = opt.theta; opt_value = opt.value;
    
    diff = 10;
    [n,~] = size(y);
    [m,~] = size(theta);
    
    T = 0;
    z = p;
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
        alpha = 2/(T+1);
        eta = T*tau/(2);
        w = (1-alpha)*p + alpha*z; 
        p_old = p; 
        ave_theta_old = ave_theta;
        
        
        param_sub.theta0 = param.theta0;
        param_sub.p0 = w;  %param.p0;
        param_sub.tol = kappa/(T^2);  % number of iter of sub-problem
        param_sub.stepsize = ell;  % stepsize of sub-problem
        param_sub.M = M;    % control the stopping criterion of sub-sub-problem
        update = diag_sub(param_sub, w, tau, y, X, lambda, a);
        gradient_count(1,T+1) = gradient_count(1,T) + update.count;
        p = update.p;
        theta = update.theta;
        
        
        [grad_p, ~] = grad(theta, w, y, X, lambda);
        z = reshape(proj_ball(z + eta*grad_p, a),n,1);
        ave_theta = (ave_theta*T*(T-1)/2 + theta*T )*2/T/(T+1);
        diff = norm(p-p_old, 2)^2 + norm(ave_theta- ave_theta_old,2)^2;
        %fprintf('the p is %d\n', p);
        %fprintf('the theta is %d\n', norm(theta));
        
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
    
    
    
    
    