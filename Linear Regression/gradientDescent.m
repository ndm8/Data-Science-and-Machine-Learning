function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    h = X*theta;     %This gives the h_theta values in a 97-length array.
                     %This is basically the prediction in our model. 

    cost_sum = 0;    %Starting with sum being 0.
    
    %%%%%%%%%%%%This updates theta_0 part%%%%%%%%%%%%%%%%
    for i = 1:m      %This calculates the summation part
        a = h(i) - y (i);
        cost_sum = cost_sum + a*(X(i, 1));
    end
    
    update = (alpha / m) * cost_sum;   %This is the - " " part
    
    temp0 = theta (1) - update;        %Updates theta and stores in temp 
                                       % variable
    
    %%%%%%%%%%%%This updates theta_1 part%%%%%%%%%%%%%%%
    %Follows the same process from theta_0
    cost_sum = 0;
    
    for i = 1:m
        a = h(i) - y (i);
        cost_sum = cost_sum + a*(X(i, 2));
    end
    
    update = (alpha / m) * cost_sum;
    
    temp1 = theta (2) - update;
    
    theta (1) = temp0;           %This actually updates theta 
    theta (2) = temp1;
    
end 
    

    
    
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

