function output = catalyst(param, sigma, beta, lambda, a, opt, method)
    tau = param.stepsize;
    x = param.x0;
    y = param.y0;
    tol = param.tol;
    ell = param.stepsize_sub; %stepsize for subproblem
    kappa = param.kappa;      %convergence parameter for subproblem
    maxT = param.maxT;
    y_opt = opt.y; x_opt = opt.x; opt_value = opt.value;
    
    
    diff = 10;
    [n,~] = size(y);
    
    T = 0;
    alpha = 1;
    sum_alpha = 0;
    v = y;
    ave_x = zeros(n,1);
    
    gradient_count = zeros(1, maxT+1);
    %  measure error
    error_value = zeros(1, maxT); error_dist = zeros(1, maxT); error_gm =zeros(1, maxT);
    error_value(1,1) = abs(loss(x, y, sigma, beta, lambda)-opt_value);
    error_dist(1,1) = norm(x - x_opt,2) + norm(y - y_opt,2);
    %error_dist(1,1) = norm(x - x_opt,2)/norm(x_opt, 2) + norm(y - y_opt,2)/norm(y_opt, 2);
    [test_grad_x1, test_grad_y1] = grad(x, y, sigma, beta, lambda);
    error_gm(1,1)=norm(y - reshape(proj(y + 0.1*test_grad_y1, a),n,1))/0.1 + norm(test_grad_x1,2);
    
    while diff>tol
        T = T+1;
        z = alpha*v +(1-alpha)*y;
        y_old = y;
        ave_x_old = ave_x;
        
        
        param_sub.x0 = x;
        param_sub.y0 = z;
        param_sub.tol = kappa*tau/(T^8);  % subproblem accuracy
        param_sub.stepsize = ell;
        if method == "EG"
           update = eg_sc(param_sub, z, tau, sigma, beta, lambda, a);
        elseif method == "GDA"
           update = gda_sc(param_sub, z, tau, sigma, beta, lambda, a);
        elseif method == "SVRG"
            param_sub.M = param.M;
            update = svrg_sub(param_sub, z, tau, sigma, beta, lambda, a);
        end
        %update.count
        gradient_count(1,T+1) = gradient_count(T) + update.count;
        y = update.y;
        x = update.x;
        
        ave_x = (ave_x*sum_alpha + x/alpha)/(sum_alpha+1/alpha);
        sum_alpha = sum_alpha + 1/alpha;
        diff = norm(y-y_old, 2)^2 + norm(ave_x- ave_x_old,2)^2;
        
        v = y_old + 1/alpha*(y-y_old);
        alpha = (-alpha^2 +sqrt(alpha^4+4*alpha^2))/2; 
        
        %measure error
        error_value(1,T+1) = abs(loss(ave_x, y, sigma, beta, lambda)-opt_value);
        error_dist(1,T+1) = norm(ave_x - x_opt,2)/norm(x_opt, 2) + norm(y - y_opt,2)/norm(y_opt, 2);
        [test_grad_x1, test_grad_y1] = grad(ave_x, y, sigma, beta, lambda);
        error_gm(1,T+1)=norm(y - reshape(proj(y + 0.1*test_grad_y1, a),n,1))/0.1 + norm(test_grad_x1,2);
        error_gm(1,T+1)
        
        if T>maxT
            fprintf('stop before converge\n');
            break;
        end
        
    end
    fprintf('converge at %d iterations\n', T);
    fprintf('converge with %d accuracy\n', diff);
    fprintf('converge with %d accuracy\n', error_dist(T+1));
    output.y = y;
    output.x = x;
    output.gradient_count = gradient_count;
    output.error_value = error_value;
    output.error_dist = error_dist;
    output.error_gm = error_gm;
    
    