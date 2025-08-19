function [GradFa] = gradientF(theta,N,x)

    individualJ = reward(theta,x);

    currentMean = mean(theta)*ones(N,1); 
    
    predictedJ = reward(currentMean,x); % predicted mean of future reward based on current belie

    GradFa = (individualJ-predictedJ)*x^2;

end