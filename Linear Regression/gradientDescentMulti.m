function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %

    
    h = X*theta;     %This gives the h_theta values in a 97-length array.
                     %This is basically the prediction in our model. 

    temp = zeros (1, size (X, 2));
    
    for j = 1:size (X, 2)
        
        cost_sum = 0;    %Starting with sum being 0.
        
        for i = 1:m      %This calculates the summation part
            a = h(i) - y (i);
            cost_sum = cost_sum + a*(X(i, j));
        end
        
        update = (alpha / m) * cost_sum;   %This is the - " " part
    
        temp (j) = theta (j) - update;        %Updates theta and stores in 
                                            % temp variable
    end 
    
    for i = 1:size (X, 2)
        theta (i) = temp (i);
    end 
    

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
