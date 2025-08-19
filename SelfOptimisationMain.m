clear all 
close all
%% Simulation Parameters
thetaStar = 1;
% theta2Star = 1;
% thetaStar = [theta1Star]
thetamin = -10;
thetamax = 10;
%% Plot reward function 
% Xmap = [0: 0.1: 2];
% Jmap = reward(thetaStar,Xmap);
% figure(1)
% plot(Xmap, Jmap);
%% Initialise Estimators 
N = 100; 
theta = thetamin + (thetamax - thetamin)*rand(N,1);

% system dynamics and gains
A = [0 1; 2 1];
B = [1; 1]; 
C = [0 1]; 

Q = [C 0;A-eye(2) B];
rank(Q)
X = Q\[1;zeros(2,1)];
Psi = X(1:2); 
G = X(3);

xi_k = [0];

P1= [0.1;0.15];

K = place(A, B, P1)

x0 = [2 1];
x_k = x0;
x_k_store =[];
x_k_store = [x_k_store; x_k];

y0 = C*x0';
y_k = y0;
y_k_store =[];
y_k_store = [y_k_store; y_k];

xi_k_store =[];
xi_k_store = [xi_k_store; xi_k];


% Learning rates

eta_0 = 0.012;
delta_0 = 0.1;  

theta_store = [];
theta_store = [theta_store theta];
theta = theta;
reward_store = [];

T = 600;
%% Iteration
for i = 1: T

    d1 = 0.0065;
    d2 = 0.003;


    eta = eta_0;
    delta = delta_0;
    %eta = 1/(1+d1*i)*eta_0; % decaying learning rate
    %delta = 1/(1+d2*i)*delta_0;


    RewardSim = reward(thetaStar,y_k);
    noise = 2*randn(size(RewardSim));
    RewardSim = RewardSim + noise;
    reward_store = [reward_store; RewardSim];

    
    % update estimators 
    Jbelief = reward(theta,y_k);
    theta = theta + eta.*(Jbelief-RewardSim*ones(N,1));

    
    theta_store = [theta_store theta];
    
    exploitation_gradient = get_exploitation_gradient(theta, N, y_k, eta);
    exploration_gradient = get_exploration_gradient(theta, N, y_k, eta);

    dual = - delta*exploitation_gradient - delta*exploration_gradient;
    %dual = - delta*exploitation_gradient - delta*exploration_gradient+0.001*rand; 
    dual = atan(dual);
    xi_k = xi_k + dual; 
        x_k = ((A-B*K)*x_k' + B*(G+K*Psi)*xi_k)';  


    y_k = C*x_k';
    xi_k_store = [xi_k_store; xi_k];
    x_k_store = [x_k_store; x_k];
    y_k_store = [y_k_store; y_k];
end


close all

set(groot, 'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');
set(0, 'defaultFigureUnits', 'centimeters', 'defaultFigurePosition', [0 0 16 9]);


thetaMean=mean(theta_store);
thetaVariance=var(theta_store);
std_dev = sqrt(thetaVariance);


figure(1)
t=1:1:T+1;
patch([t fliplr(t)], [thetaMean-std_dev  fliplr(thetaMean+std_dev)], [0.3010, 0.7450, 0.9330])
hold on
plot(1:1:T+1,thetaMean,'LineWidth',1.5, 'Color', '[0.4940, 0.1840, 0.5560]');
plot(1:1:T,ones(T,1), 'Color', 'r', 'LineWidth',1.5,'LineStyle', '--')
box on


xlim([0,T]);
my_legend = legend('standard deviation of $\theta$','estimated mean of $\theta$', 'optimal $\theta^*$');
set(my_legend,'FontSize',12);
xlabel('$k$','FontSize',12);
ylabel('$\theta$','FontSize',12);


figure(2)
plot(1:1:T+1,y_k_store(:,1),'Color', 'k', 'LineWidth',1.5); hold on;
plot(1:1:T,1*ones(T,1),'Color', 'r', 'LineWidth',1.5,'LineStyle', '--')


xlim([0,T]);
my_legend = legend('output $y$','optimal reference');
set(my_legend,'FontSize',12);
xlabel('$k$','FontSize',12);
ylabel('$y$','FontSize',12);


figure(3)
plot(1:1:T+1,x_k_store(:,1:2),'LineWidth',1.5); hold on;


xlim([0,T]);
my_legend = legend('state $x_1$','state $x_2$');
set(my_legend,'FontSize',12);
xlabel('$k$','FontSize',12);
ylabel('$x$','FontSize',12);


figure(4)
plot(1:1:T,reward_store,'Color', '[0, 0.4470, 0.7410]', 'LineWidth',1.5); hold on;
plot(1:1:T,1*ones(T,1),'Color', 'r', 'LineWidth',1.5,'LineStyle', '--')


xlim([0,T]);
my_legend = legend('observed reward $J$', 'optimal reward','Location','southeast');
set(my_legend,'FontSize',12);
xlabel('$k$','FontSize',12);
ylabel('$J$','FontSize',12);