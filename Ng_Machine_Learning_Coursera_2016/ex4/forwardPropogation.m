function [a1, z2, a2, z3, a3, J] = forwardPropogation(X,y,Theta1, Theta2, num_labels, input_layer_size, hidden_layer_size, lambda)
%%%%%%%%%%%%%%  COST FUNCTION %%%%%%%%%%%%%%
% check input dimensions
% Theta1 [25 x 401] = [hidden layer size x input size]
% Theta2 [10 x 26] = [hidden layer size x input size]
% X = [5000 x 400] = [num data points x input size]
% y = [5000 x 1] = [num data points x 1]
% new_Y = [5000 x 10] = [num data points x num_classes]


% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;


% PART 1
%
y_matrix = eye(num_labels)(y,:); 


% FEEDFORWARD FUNCTION (run the neural network)
% Theta1: [25 x 401] = [hidden layer size x input size]
% Theta2: [10 x 26] = [hidden layer size x input size]
% X: [5000 x 400] = [num data points x input size]
% y: [5000 x 1] = [num data points x 1]
% new_Y: [5000 x 10] = [num data points x num_classes]
% a1: [5000 x 401]
a1 = [ones(length(X),1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(length(a2),1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
h = a3;

% UNREGULARIZED COST FUNCTION
K = num_labels;
for i = 1:m
  for k = 1:K
    J += (-y_matrix(i,k) * log((h(i,k))) - (1 - y_matrix(i,k)) * log(1 - h(i,k)));
  end
end

J = J/m;

% REGULARIZATION TERM
reg_term = 0;

Theta1(:,1) = 0;
Theta2(:,1) = 0;

% compute the regularization term (see pg. 6 of ex4)
reg_term += sum(sum(Theta1 .* Theta1));
reg_term += sum(sum(Theta2 .* Theta2));


reg_term = reg_term * (lambda/(2*m));


J += reg_term;

end