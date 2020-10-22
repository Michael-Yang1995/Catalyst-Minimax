function output = diagoh(param, sigma, beta, lambda, a, opt)
    tau = param.stepsize;
    x = param.x0;
    y = param.y0;
    tol = param.tol;
    ell = param.stepsize_sub; %stepsize for subproblem
    kappa = param.kappa;      %convergence parameter for subproblem
    M = param.M;         %convergence parameter for subproblem' subproblem
    maxT = param.maxT;
    x_opt = opt.x; y_opt = opt.y; opt_value = opt.value;
    
    diff = 10;
    [m,~] = size(y); n = m;

    
    T = 0;
    z = y;
    ave_x = zeros(m,1);
    
    gradient_count = zeros(1, maxT+1);
    %  measure error
    error_value = zeros(1, maxT); error_dist = zeros(1, maxT); error_gm =zeros(1, maxT);
    error_value(1,1) = abs(loss(x, y, sigma, beta, lambda)-opt_value);
    error_dist(1,1) = norm(x - x_opt,2) + norm(y - y_opt,2);
    %error_dist(1,1) = norm(x - x_opt,2)/norm(x_opt, 2) + norm(y - y_opt,2)/norm(y_opt,2);
    [test_grad_x1, test_grad_y1] = grad(x, y, sigma, beta, lambda);
    error_gm(1,1)=norm(y - reshape(proj(y + 0.1*test_grad_y1, a),n,1))/0.1 + norm(test_grad_x1,2);
    
    while diff>tol
        T = T+1;
        alpha = 2/(T+1);
        eta = T*tau/(2);
        w = (1-alpha)*y + alpha*z; 
        y_old = y; 
        ave_x_old = ave_x;
        
        
        param_sub.x0 = param.x0; %x;
        param_sub.y0 = w;
        param_sub.tol = kappa/(T^2);  % number of iter of sub-problem
        param_sub.stepsize = ell;  % stepsize of sub-problem
        param_sub.M = M;    % control the stopping criterion of sub-sub-problem
        update = diag_sub(param_sub, w, tau, sigma, beta, lambda, a);
        gradient_count(1,T+1) = gradient_count(1,T) + update.count;
        x = update.x;
        y = update.y;
        
        
        [~, grad_y] = grad(x, w, sigma, beta, lambda);
        z = reshape(proj(z + eta*grad_y, a),n,1);
        ave_x = (ave_x*T*(T-1)/2 + x*T )*2/T/(T+1);
        diff = norm(y-y_old, 2)^2 + norm(ave_x- ave_x_old,2)^2;
        %fprintf('the p is %d\n', p);
        %fprintf('the theta is %d\n', norm(theta));
        
        %measure error
        error_value(1,T+1) = abs(loss(ave_x, y, sigma, beta, lambda)-opt_value);
        error_dist(1,T+1) = norm(ave_x - x_opt,2) + norm(y - y_opt,2);
        %error_dist(1,T+1) = norm(ave_x - x_opt,2)/norm(x_opt, 2) + norm(y - y_opt,2)/norm(y_opt, 2);
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
    output.y = y;
    output.x = ave_x;
    output.gradient_count = gradient_count;
    output.error_value = error_value;
    output.error_dist = error_dist;
    output.error_gm = error_gm;
    
    
    
    
    