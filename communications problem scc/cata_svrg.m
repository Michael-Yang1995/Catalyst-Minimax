function output = cata_svrg(param, y, X, lambda, a, opt)
    tau = param.stepsize;
    theta = param.theta0;
    p = param.p0;
    tol = param.tol;
    ell = param.stepsize_sub; %stepsize for subproblem
    kappa = param.kappa;      %convergence parameter for subproblem
    maxT = param.maxT;
    M = param.M;
    p_opt = opt.p; theta_opt = opt.theta; opt_value = opt.value;
    
    
    diff = 10;
    [n,~] = size(y);
    [m,~] = size(theta);
    
    T = 0;
    alpha = 1;
    sum_alpha = 0;
    v = p;
    ave_theta = zeros(m,1);
    
    gradient_count = zeros(1, maxT+1);
    %  measure error
    error_value = zeros(1, maxT); error_dist = zeros(1, maxT);
    error_value(1,1) = abs(loss(theta, p, y, X, lambda)-opt_value);
    error_dist(1,1) = norm(theta - theta_opt,2) + norm(p - p_opt,2);
    
    while diff>tol
        T = T+1;
        z = alpha*v +(1-alpha)*p;
        p_old = p;
        ave_theta_old = ave_theta;
        
        %subroutine -- svrg_sub
        param_sub.theta0 = theta;
        param_sub.p0 = z;
        param_sub.tol = kappa*tau/(T^4);   %/T^12
        param_sub.stepsize = ell;
        param_sub.M = M;
        update = svrg_sub(param_sub, z, tau, y, X, lambda, a);
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
        %error_dist(1,T+1) = norm(ave_theta - theta_opt,2) + norm(p - p_opt,2);
        error_dist(1,T+1) = norm(theta - theta_opt,2) + norm(p - p_opt,2);
        
        
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
    output.error_value = error_value;
    output.error_dist = error_dist;
    