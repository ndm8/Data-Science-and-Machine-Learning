function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


z = X * theta;
h = sigmoid (z);

cost_sum = 0;
for i = 1:m
 
    a = -y(i)*log(h(i));
    b = (1 - y(i))*log (1 - h (i));
    cost_sum = cost_sum + (a - b);
end

J = cost_sum / m; 

for j = 1:size (grad)
    pd_sum = 0;
    for i = 1:m
        a = (h(i) - y (i)) * X (i, j);
        pd_sum = pd_sum + a;
    end
    grad (j) = pd_sum / m;
end




% =============================================================

end
