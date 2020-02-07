function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


h = X*theta;     %This gives the h_theta values in a 97-length array.

sum = 0;

cost_sum = 0;
for i = 1:m
    a = h(i) - y (i);
    cost_sum = cost_sum + a^2;
end

J = cost_sum/(2*m);

% =========================================================================

end
