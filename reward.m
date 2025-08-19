function [ J ] = reward(theta,y)

theta1 = theta(:,1);

J = 2*y - theta1*y^2; 
end