function exploration_gradient = get_exploration_gradient(theta, N, x, eta)

delta_theta = 0.001;


[GradFa] = gradientF(theta,N,x); 

future_thetaA = theta-eta.*GradFa;

future_r = 1./future_thetaA;

future_r_mean = mean(future_r)*ones(N,1);

exploration_mean = ((future_r-future_r_mean)'*(future_r-future_r_mean));


[GradFa_delta] = gradientF(theta+delta_theta,N,x); 

future_thetaA_delta = theta+delta_theta-eta.*GradFa_delta;

future_r_delta = 1./future_thetaA_delta;

future_r_delta_mean = mean(future_r_delta)*ones(N,1);

exploration_delta_mean = ((future_r_delta-future_r_delta_mean)'*(future_r_delta-future_r_delta_mean));


exploration_gradient = (exploration_delta_mean-exploration_mean)./delta_theta;

end