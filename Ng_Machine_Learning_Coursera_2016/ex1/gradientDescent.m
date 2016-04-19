function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
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

%  simultaneously:    
%  d/dtheta_0 J(theta_0, theta_1) = 1/m (sum_{i = 1}^M (h_theta(x^(i) - y^(i)))
%  d/dtheta_1 J(theta_0, theta_1) = 1/m (sum_{i = 1}^M (h_theta(x^(i) - y^(i)) x^(i))
%  theta_0 = theta(0) - alpha * d/dtheta_0 * J (theta_0, theta_1)
%  theta_1 = theta(1) - alpha * d/dtheta_1 * J (theta_0, theta_1)

% therefore:
% theta_0 = theta_0 - alpha 1/m sum_{i=1}^m (h_{theta(x^{(i)})} - y^{(i)})
% theta_1 = theta_0 - alpha 1/m sum_{i=1}^m ((h_{theta(x^{(i)})} - y^{(i)})(x^{(i)})

% get a vector of values for this value of theta_0, theta_1 and all X values
h = X*theta;
theta = theta - alpha * (1/m) * X'*(h - y);

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
