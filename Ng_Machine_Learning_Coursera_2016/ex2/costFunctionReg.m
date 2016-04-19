function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
n = size(theta);


h = sigmoid(X * theta);
log_h = log(h);
log_1_minus_h = log(1 - h);
one_minus_y = 1 - y;




h_minus_y = h - y;

% vector versions of J equation and gradient dJ_d_theta:

grad = (1/m) * X' * h_minus_y;

for j = 2:n
  grad(j) += (lambda/m) * theta(j);
end

theta(1) = 0;
J = (-1/m) * ((log_h)'*y + log_1_minus_h'*(1-y)) + \
    lambda/(2*m) * theta' * theta;

% =============================================================

end
