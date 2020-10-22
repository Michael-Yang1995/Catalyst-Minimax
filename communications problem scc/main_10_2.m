rng(5);

n = 500;
beta = ones(n, 1);
sigma = rand(n,1)*10;
a = n;
x0 = rand(n,1)*5;
y0 = rand(n,1);
lambda = 0.01;
opt.x = zeros(n,1); opt.y = zeros(n,1); opt.value = 0;




%% find optimal 


param_opt.y0 = opt.y;
param_opt.x0 = opt.x;
param_opt.tol = 10^(-30);    
param_opt.stepsize = 2;
param_opt.stepsize_sub = 65;    % sbuproblem stepsize 
param_opt.kappa = 0.01;     % kappa is used to tune subproblem stropping criterion
param_opt.M = 0.001; 
param_opt.maxT = 30000;

output_opt = diagoh(param_opt, sigma, beta, lambda, a, opt);

opt.x = output_opt.x; opt.y = output_opt.y;
opt.value = loss(opt.x, opt.y, sigma, beta, lambda);




%% extragradient 

%opt.x = zeros(n,1); opt.y = zeros(n,1); opt.value = 0;
param_eg.x0 = x0;
param_eg.y0 = y0;
%param_eg.x0 = output_eg.x;
%param_eg.y0 = output_eg.y;
param_eg.tol = 10^(-30);
param_eg.stepsize = 2.2;
param_eg.maxT = 12500;


output_eg = extragrad(param_eg, sigma, beta, lambda, a, opt);

output_eg.gradient_count = 0:4:(4*param_eg.maxT);






%% Catalyst-extragradient

param_catalyst.y0 = y0;
param_catalyst.x0 = x0;
param_catalyst.tol = 10^(-30);    
param_catalyst.stepsize = 0.08;   % 1 AND 0.08
param_catalyst.stepsize_sub = 2.2;    % subproblem stepsize 
param_catalyst.kappa = 0.0001;     % kappa is used to tune subproblem stropping criterion
param_catalyst.maxT = 12000; 

output_catalyst = catalyst(param_catalyst,sigma, beta, lambda, a, opt, "EG");

gradient_count_catalyst = output_catalyst.gradient_count;





%% diag


param_diag.x0 = x0;
param_diag.y0 = y0;
param_diag.tol = 10^(-22);
param_diag.stepsize = 2;
param_diag.stepsize_sub = 65;    % sbuproblem stepsize 
param_diag.kappa = 0.01;     % kappa is used to tune subproblem stropping criterion
param_diag.M = 0.001; 
param_diag.maxT = 2000;

output_diag = diagoh(param_diag, sigma, beta, lambda, a, opt);














%%  plot distance error

MS = 16;
LW = 4;
LegFont = 12;
%iter = 100;


figure
axes('XScale', 'linear', 'YScale', 'log');
hold on
eg_errors = loglog(output_eg.gradient_count(1:12500), output_eg.error_dist(1:12500), 'b-', 'LineWidth', LW);
catalyst_errors = loglog(output_catalyst.gradient_count(1:12000), output_catalyst.error_dist(1:12000), 'r-', 'LineWidth', LW);
diag_errors = loglog(output_diag.gradient_count(1:1500), output_diag.error_dist(1:1500), 'color',[0, 0.5, 0], 'LineWidth', LW);

hold off
set(gca, 'fontsize', 18)
l = legend([ eg_errors,catalyst_errors, diag_errors], 'EG', 'Catalyst-EG','DIAG', 'Location','northeast');
xlabel('Gradient Count', 'fontsize', 32)
ylabel('$\Vert x_t-x^*\Vert+\Vert y_t-y^*\Vert$','Interpreter','latex', 'fontsize', 32)
xlim([0 50000])
ylim([10^(-10) 1000])
set(gca, 'FontSize', 50)
grid on


%% plot gradient norm
MS = 16;
LW = 4;
LegFont = 12;


figure
axes('XScale', 'linear', 'YScale', 'log');
hold on
eg_errors = loglog(output_eg.gradient_count(1:12500), output_eg.error_gm(1:12500), 'b-', 'LineWidth', LW);
catalyst_errors = loglog(output_catalyst.gradient_count(1:12000), output_catalyst.error_gm(1:12000), 'r-', 'LineWidth', LW);
diag_errors = loglog(output_diag.gradient_count(1:1500), output_diag.error_gm(1:1500), 'color',[0, 0.5, 0], 'LineWidth', LW);

hold off
set(gca, 'fontsize', 18)
l = legend([ eg_errors,catalyst_errors, diag_errors], 'EG', 'Catalyst-EG','DIAG', 'Location','northeast');
xlabel('Gradient Count', 'fontsize', 32)
ylabel('Gradient Mapping', 'fontsize', 32)
xlim([0 50000])
ylim([10^(-10) 1000])
set(gca, 'FontSize', 50)
grid on

%%
param_eg.x0 = x0;
param_eg.y0 = y0;
param_eg.tol = 10^(-30);
param_eg.stepsize = 1;
param_eg.maxT = 12500;
output_eg1 = extragrad(param_eg, sigma, beta, lambda, a, opt);
output_eg1.gradient_count = 0:4:(4*param_eg.maxT);


param_catalyst.y0 = y0;
param_catalyst.x0 = x0;
param_catalyst.tol = 10^(-30);    
param_catalyst.stepsize = 0.08;   % 1 AND 0.08
param_catalyst.stepsize_sub = 1;    % subproblem stepsize 
param_catalyst.kappa = 0.0001;     % kappa is used to tune subproblem stropping criterion
param_catalyst.maxT = 12000; 
output_catalyst1 = catalyst(param_catalyst,sigma, beta, lambda, a, opt, "EG");
gradient_count_catalyst1 = output_catalyst.gradient_count;


MS = 16;
LW = 2;
LegFont = 12;
iter = 100;


figure
axes('XScale', 'linear', 'YScale', 'log');
hold on
eg_errors = loglog(output_eg1.gradient_count(1:12500), output_eg1.error_dist(1:12500), 'b-', 'LineWidth', LW);
catalyst_errors = loglog(output_catalyst1.gradient_count(1:12000), output_catalyst1.error_dist(1:12000), 'r-', 'LineWidth', LW);
hold off
set(gca, 'fontsize', 18)
l = legend([ eg_errors,catalyst_errors], 'EG', 'Catalyst-EG', 'Location','northeast');
l.FontSize = LegFont;
 %title('(b) Convergence of deterministic GDA', 'fontsize', 18)
xlabel('# Oracles', 'fontsize', 24)
ylabel('$\Vert x_t-x^*\Vert+\Vert y_t-y^*\Vert$','Interpreter','latex', 'fontsize', 24)


%%
param_eg.x0 = x0;
param_eg.y0 = y0;
param_eg.tol = 10^(-30);
param_eg.stepsize = 1.5;
param_eg.maxT = 12500;
output_eg2 = extragrad(param_eg, sigma, beta, lambda, a, opt);
output_eg2.gradient_count = 0:4:(4*param_eg.maxT);


param_catalyst.y0 = y0;
param_catalyst.x0 = x0;
param_catalyst.tol = 10^(-30);    
param_catalyst.stepsize = 0.08;   % 1 AND 0.08
param_catalyst.stepsize_sub = 1.5;    % subproblem stepsize 
param_catalyst.kappa = 0.0001;     % kappa is used to tune subproblem stropping criterion
param_catalyst.maxT = 12000; 
output_catalyst2 = catalyst(param_catalyst,sigma, beta, lambda, a, opt, "EG");
gradient_count_catalyst2 = output_catalyst.gradient_count;


MS = 16;
LW = 2;
LegFont = 12;
iter = 100;


%%
figure
axes('XScale', 'linear', 'YScale', 'log');
hold on
eg_errors = loglog(output_eg2.gradient_count(1:12500), output_eg2.error_dist(1:12500), 'b-', 'LineWidth', LW);
catalyst_errors = loglog(output_catalyst2.gradient_count(1:12000), output_catalyst2.error_dist(1:12000), 'r-', 'LineWidth', LW);
hold off
set(gca, 'fontsize', 18)
l = legend([ eg_errors,catalyst_errors], 'EG', 'Catalyst-EG', 'Location','northeast');
l.FontSize = LegFont;
 %title('(b) Convergence of deterministic GDA', 'fontsize', 18)
xlabel('# Oracles', 'fontsize', 24)
ylabel('$\Vert x_t-x^*\Vert+\Vert y_t-y^*\Vert$','Interpreter','latex', 'fontsize', 24)


%%
param_eg.x0 = x0;
param_eg.y0 = y0;
param_eg.tol = 10^(-30);
param_eg.stepsize = 2;
param_eg.maxT = 12500;
output_eg3 = extragrad(param_eg, sigma, beta, lambda, a, opt);
output_eg3.gradient_count = 0:4:(4*param_eg.maxT);


param_catalyst.y0 = y0;
param_catalyst.x0 = x0;
param_catalyst.tol = 10^(-30);    
param_catalyst.stepsize = 0.08;   % 1 AND 0.08
param_catalyst.stepsize_sub = 2;    % subproblem stepsize 
param_catalyst.kappa = 0.0001;     % kappa is used to tune subproblem stropping criterion
param_catalyst.maxT = 12000; 
output_catalyst3 = catalyst(param_catalyst,sigma, beta, lambda, a, opt, "EG");
gradient_count_catalyst3 = output_catalyst.gradient_count;



%%

figure
axes('XScale', 'linear', 'YScale', 'log');
hold on
eg_errors = loglog(output_eg3.gradient_count(1:12500), output_eg3.error_dist(1:12500), 'b-', 'LineWidth', LW);
catalyst_errors = loglog(output_catalyst3.gradient_count(1:12000), output_catalyst3.error_dist(1:12000), 'r-', 'LineWidth', LW);
hold off
set(gca, 'fontsize', 18)
l = legend([ eg_errors,catalyst_errors], 'EG', 'Catalyst-EG', 'Location','northeast');
l.FontSize = LegFont;
 %title('(b) Convergence of deterministic GDA', 'fontsize', 18)
xlabel('# Oracles', 'fontsize', 24)
ylabel('$\Vert x_t-x^*\Vert+\Vert y_t-y^*\Vert$','Interpreter','latex', 'fontsize', 24)

%%

MS = 16;
LW = 4;
LegFont = 12;
figure
axes('XScale', 'linear', 'YScale', 'log');
hold on
eg_errors1 = loglog(output_eg1.gradient_count(1:12500), output_eg1.error_dist(1:12500), 'b-', 'LineWidth', LW);
catalyst_errors1 = loglog(output_catalyst3.gradient_count(1:12000), output_catalyst3.error_dist(1:12000), 'r-', 'LineWidth', LW);
eg_errors2 = loglog(output_eg2.gradient_count(1:12500), output_eg2.error_dist(1:12500), 'b--', 'LineWidth', LW);
catalyst_errors2 = loglog(output_catalyst1.gradient_count(1:12000), output_catalyst1.error_dist(1:12000), 'r--', 'LineWidth', LW);
eg_errors3 = loglog(output_eg3.gradient_count(1:12500), output_eg3.error_dist(1:12500), 'b:', 'LineWidth', LW);
catalyst_errors3 = loglog(output_catalyst2.gradient_count(1:12000), output_catalyst2.error_dist(1:12000), 'r:', 'LineWidth', LW);
hold off
set(gca, 'fontsize', 18)
l = legend([ eg_errors1,catalyst_errors1,eg_errors2,catalyst_errors2,eg_errors3,catalyst_errors3], 'EG w/ stepsize 1', 'Catalyst-EG w/ stepsize 1','EG w/ stepsize 1.5', 'Catalyst-EG w/ stepsize 1.5','EG w/ stepsize 2', 'Catalyst-EG w/ stepsize 2', 'Location','northeast');
 %title('(b) Convergence of deterministic GDA', 'fontsize', 18)
xlabel('Gradient Count', 'fontsize', 24)
ylabel('$\Vert x_t-x^*\Vert+\Vert y_t-y^*\Vert$','Interpreter','latex', 'fontsize', 24)
xlim([0 20000])
ylim([10^(-8) 1000])
set(gca, 'FontSize', 50)
grid on
