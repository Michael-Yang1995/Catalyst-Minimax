data1 = readtable('WDBC.dat');
X0 = data1{:,3:32};
y1 = data1(:,2);
y2 = zeros(569,1);
for i=1:569
    if y1{i,1}=="B"
        y2(i)=-1;
    else
        y2(i) = 1;
    end
end

%y is the label and X is features

%% rescale of X matrix
X1 = X0-mean(X0);
X2 = X1./repmat(std(X1),569,1);

%% seperate into training and testing data 
rng(5);
test_size = floor(569*0.2);
idx = randsample(569, test_size);
X_test = X2(idx,:);
y_test = y2(idx,:);
X = X2(setdiff(1:569, idx),:);
y = y2(setdiff(1:569, idx),:);



rng(5);

n = 569-test_size;
m = 30;
lambda = 0.6;
a = 1/200;
p0 = randn(n,1);
theta0 = randn(m, 1);
%opt.p = opt_p; opt.theta = opt_theta; opt.value = opt_value;

load('opt.mat')

%% extragradient 

param_eg.p0 = p0;
param_eg.theta0 = theta0;
param_eg.tol = 10^(-20);
param_eg.stepsize = 0.55;
param_eg.maxT = 5000;


output_eg = extragrad(param_eg, y, X, lambda, a, opt, y_test, X_test);

output_eg.gradient_count = 0:4:(4*param_eg.maxT);





%% Catalyst-extragradient

param_catalyst.p0 = abs(p0);
param_catalyst.theta0 = theta0;
param_catalyst.tol = 10^(-50);
param_catalyst.stepsize = 10;
param_catalyst.stepsize_sub = 0.3;    % sbuproblem stepsize 
param_catalyst.kappa = 0.00001;     % kappa is used to tune subproblem stopping criterion
param_catalyst.maxT = 10000;

output_catalyst = catalyst(param_catalyst, y, X, lambda, a, opt, y_test, X_test);

gradient_count_catalyst = output_catalyst.gradient_count;


%% diag


param_diag.p0 = abs(p0);
param_diag.theta0 = theta0;
param_diag.tol = 10^(-22);
param_diag.stepsize = 200;
param_diag.stepsize_sub = 0.5;    % sbuproblem stepsize 
param_diag.kappa = 0.01;     % kappa is used to tune subproblem stropping criterion
param_diag.M = 0.001; 
param_diag.maxT = 160;

output_diag = diagoh(param_diag, y, X, lambda, a, opt, y_test, X_test);











%% plot distance to limit point
MS = 16;
LW = 4;
LegFont = 12;

figure
axes('XScale', 'linear', 'YScale', 'log');
hold on
eg_errors = loglog(output_eg.gradient_count(1:5000), output_eg.error_dist(1:5000), 'b-', 'LineWidth', LW);
catalyst_errors = loglog(output_catalyst.gradient_count(1:10000), output_catalyst.error_dist(1:10000), 'r-', 'LineWidth', LW);
diag_errors = loglog(output_diag.gradient_count(1:160), output_diag.error_dist(1:160), 'color',[0, 0.5, 0], 'LineWidth', LW);
hold off
set(gca, 'fontsize', 18)
l = legend([ eg_errors,catalyst_errors, diag_errors], 'EG', 'Catalyst-EG','DIAG', 'Location','northeast');
xlabel('Gradient Count', 'fontsize', 24)
ylabel('$\Vert x_t-x^*\Vert+\Vert y_t-y^*\Vert$','Interpreter','latex', 'fontsize', 24)
xlim([0 20000])
ylim([10^(-8) 1000])
set(gca, 'FontSize', 50)
grid on

%% plot gradient mapping
MS = 16;
LW = 4;
LegFont = 12;

figure
axes('XScale', 'linear', 'YScale', 'log');
hold on
eg_errors = loglog(output_eg.gradient_count(1:5000), output_eg.error_gm(1:5000), 'b-', 'LineWidth', LW);
catalyst_errors = loglog(output_catalyst.gradient_count(1:10000), output_catalyst.error_gm(1:10000), 'r-', 'LineWidth', LW);
diag_errors = loglog(output_diag.gradient_count(1:160), output_diag.error_gm(1:160), 'color',[0, 0.5, 0], 'LineWidth', LW);
hold off
set(gca, 'fontsize', 18)
l = legend([ eg_errors,catalyst_errors, diag_errors], 'EG', 'Catalyst-EG','DIAG', 'Location','northeast');
xlabel('Gradient Count', 'fontsize', 32)
ylabel('Gradient Mapping', 'fontsize', 32)
xlim([0 20000])
ylim([10^(-8) 1000])
set(gca, 'FontSize', 50)
grid on


%%  plot distance error

MS = 16;
LW = 2;
LegFont = 12;
iter = 100;


figure
axes('XScale', 'linear', 'YScale', 'log');
hold on
eg_errors = loglog(output_eg.gradient_count(1:5000), output_eg.error_value(1:5000), 'b-', 'LineWidth', LW);
catalyst_errors = loglog(output_catalyst.gradient_count(1:9990), output_catalyst.error_value(1:9990), 'r-', 'LineWidth', LW);
diag_errors = loglog(output_diag.gradient_count(1:160), output_diag.error_value(1:160), 'c-', 'LineWidth', LW);
hold off
set(gca, 'fontsize', 18)
l = legend([ eg_errors,catalyst_errors, diag_errors], 'extra-gradient', 'Catalyst-EG','DIAG', 'Location','northeast');
l.FontSize = LegFont;
xlabel('# Oracles', 'fontsize', 24)
ylabel('$|f(x,y) - f(x^*, y^*)|$','Interpreter','latex', 'fontsize', 24)



%%
param_eg.p0 = p0;
param_eg.theta0 = theta0;
param_eg.tol = 10^(-20);
param_eg.stepsize = 0.2;
param_eg.maxT = 5000;
output_eg1 = extragrad(param_eg, y, X, lambda, a, opt, y_test, X_test);
output_eg1.gradient_count = 0:4:(4*param_eg.maxT);


param_catalyst.p0 = abs(p0);
param_catalyst.theta0 = theta0;
param_catalyst.tol = 10^(-50);
param_catalyst.stepsize = 10;
param_catalyst.stepsize_sub = 0.2;    % sbuproblem stepsize 
param_catalyst.kappa = 0.00001;     % kappa is used to tune subproblem stopping criterion
param_catalyst.maxT = 10000;
output_catalyst1 = catalyst(param_catalyst, y, X, lambda, a, opt, y_test, X_test);
%gradient_count_catalyst1 = output_catalyst1.gradient_count;

%%
MS = 16;
LW = 2;
LegFont = 12;
iter = 100;


figure
axes('XScale', 'linear', 'YScale', 'log');
hold on
eg_errors = loglog(output_eg1.gradient_count(1:5000), output_eg1.error_dist(1:5000), 'b-', 'LineWidth', LW);
catalyst_errors = loglog(output_catalyst1.gradient_count(1:9990), output_catalyst1.error_dist(1:9990), 'r-', 'LineWidth', LW);
hold off
set(gca, 'fontsize', 18)
l = legend([ eg_errors,catalyst_errors], 'EG', 'Catalyst-EG', 'Location','northeast');
l.FontSize = LegFont;
xlabel('# Oracles', 'fontsize', 24)
ylabel('$\Vert x_t-x^*\Vert+\Vert y_t-y^*\Vert$','Interpreter','latex', 'fontsize', 24)


%%
param_eg.p0 = p0;
param_eg.theta0 = theta0;
param_eg.tol = 10^(-20);
param_eg.stepsize = 0.3;
param_eg.maxT = 5000;
output_eg2 = extragrad(param_eg, y, X, lambda, a, opt, y_test, X_test);
output_eg2.gradient_count = 0:4:(4*param_eg.maxT);


param_catalyst.p0 = abs(p0);
param_catalyst.theta0 = theta0;
param_catalyst.tol = 10^(-50);
param_catalyst.stepsize = 10;
param_catalyst.stepsize_sub = 0.3;    % sbuproblem stepsize 
param_catalyst.kappa = 0.00001;     % kappa is used to tune subproblem stopping criterion
param_catalyst.maxT = 10000;
output_catalyst2 = catalyst(param_catalyst, y, X, lambda, a, opt, y_test, X_test);
%gradient_count_catalyst1 = output_catalyst1.gradient_count;

%%
MS = 16;
LW = 2;
LegFont = 12;
iter = 100;


figure
axes('XScale', 'linear', 'YScale', 'log');
hold on
eg_errors = loglog(output_eg2.gradient_count(1:5000), output_eg2.error_dist(1:5000), 'b-', 'LineWidth', LW);
catalyst_errors = loglog(output_catalyst2.gradient_count(1:9990), output_catalyst2.error_dist(1:9990), 'r-', 'LineWidth', LW);
hold off
set(gca, 'fontsize', 18)
l = legend([ eg_errors,catalyst_errors], 'EG', 'Catalyst-EG', 'Location','northeast');
l.FontSize = LegFont;
xlabel('# Oracles', 'fontsize', 24)
ylabel('$\Vert x_t-x^*\Vert+\Vert y_t-y^*\Vert$','Interpreter','latex', 'fontsize', 24)



%%
param_eg.p0 = p0;
param_eg.theta0 = theta0;
param_eg.tol = 10^(-20);
param_eg.stepsize = 0.4;
param_eg.maxT = 5000;
output_eg3 = extragrad(param_eg, y, X, lambda, a, opt, y_test, X_test);
output_eg3.gradient_count = 0:4:(4*param_eg.maxT);


param_catalyst.p0 = abs(p0);
param_catalyst.theta0 = theta0;
param_catalyst.tol = 10^(-50);
param_catalyst.stepsize = 10;
param_catalyst.stepsize_sub = 0.4;    % sbuproblem stepsize 
param_catalyst.kappa = 0.00001;     % kappa is used to tune subproblem stopping criterion
param_catalyst.maxT = 10000;
output_catalyst3 = catalyst(param_catalyst, y, X, lambda, a, opt, y_test, X_test);
%gradient_count_catalyst1 = output_catalyst1.gradient_count;

%%
MS = 16;
LW = 2;
LegFont = 12;
iter = 100;


figure
axes('XScale', 'linear', 'YScale', 'log');
hold on
eg_errors = loglog(output_eg3.gradient_count(1:5000), output_eg3.error_dist(1:5000), 'b-', 'LineWidth', LW);
catalyst_errors = loglog(output_catalyst3.gradient_count(1:9990), output_catalyst3.error_dist(1:9990), 'r-', 'LineWidth', LW);
hold off
set(gca, 'fontsize', 18)
l = legend([ eg_errors,catalyst_errors], 'EG', 'Catalyst-EG', 'Location','northeast');
l.FontSize = LegFont;
xlabel('# Oracles', 'fontsize', 24)
ylabel('$\Vert x_t-x^*\Vert+\Vert y_t-y^*\Vert$','Interpreter','latex', 'fontsize', 24)



%%

MS = 16;
LW = 4;
LegFont = 12;
iter = 100;


figure
axes('XScale', 'linear', 'YScale', 'log');
hold on
eg_errors1 = loglog(output_eg1.gradient_count(1:5000), output_eg1.error_dist(1:5000), 'b-', 'LineWidth', LW);
catalyst_errors1 = loglog(output_catalyst3.gradient_count(1:9990), output_catalyst3.error_dist(1:9990), 'r-', 'LineWidth', LW);
eg_errors2 = loglog(output_eg2.gradient_count(1:5000), output_eg2.error_dist(1:5000), 'b--', 'LineWidth', LW);
catalyst_errors2 = loglog(output_catalyst1.gradient_count(1:9990), output_catalyst1.error_dist(1:9990), 'r--', 'LineWidth', LW);
eg_errors3 = loglog(output_eg3.gradient_count(1:5000), output_eg3.error_dist(1:5000), 'b:', 'LineWidth', LW);
catalyst_errors3 = loglog(output_catalyst2.gradient_count(1:9990), output_catalyst2.error_dist(1:9990), 'r:', 'LineWidth', LW);
hold off
set(gca, 'fontsize', 18)
l = legend([ eg_errors1,catalyst_errors1,eg_errors2,catalyst_errors2,eg_errors3,catalyst_errors3], 'EG w/ stepsize .2', 'Catalyst-EG w/ stepsize .2','EG w/ stepsize .3', 'Catalyst-EG w/ stepsize .3','EG w/ stepsize .4', 'Catalyst-EG w/ stepsize .4', 'Location','northeast');
xlabel('Gradient Count', 'fontsize', 24)
ylabel('$\Vert x_t-x^*\Vert+\Vert y_t-y^*\Vert$','Interpreter','latex', 'fontsize', 24)
xlim([0 20000])
ylim([10^(-8) 1000])
set(gca, 'FontSize', 50)
grid on



