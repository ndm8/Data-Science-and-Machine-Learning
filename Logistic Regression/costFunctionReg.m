function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length (theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = X * theta;
h = sigmoid (z);

cost_sum = 0;
for i = 1:m
 
    a = -y(i)*log(h(i));
    b = (1 - y(i))*log (1 - h (i));
    cost_sum = cost_sum + (a - b);
end

theta_sum = 0;
for j = 1:n
    a = theta (j);
    theta_sum = theta_sum + a^2;
end

lambda_part = theta_sum * (lambda / (2 * m));

J = (cost_sum / m) + lambda_part; 

pd_sum = 0;
for i = 1:m
    a = (h(i) - y (i)) * X (i, 1);
    pd_sum = pd_sum + a;
end
grad (1) = pd_sum / m;

for j = 2:size (grad)
    pd_sum = 0;
    for i = 1:m
        a = (h(i) - y (i)) * X (i, j);
        pd_sum = pd_sum + a;
    end
    grad (j) = (pd_sum / m) + (lambda / m) * theta (j);
end



% =============================================================

end
