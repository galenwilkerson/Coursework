function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

                 
%%%%%%%%%%%%%%  COST FUNCTION %%%%%%%%%%%%%%

[a1, z2, a2, z3, a3, J] = forwardPropogation(X,y,Theta1, Theta2, num_labels, input_layer_size, hidden_layer_size, lambda);


% -------------------------------------------------------------

% PART 2 - BACKPROPAGATION

%m = the number of training examples
%n = the number of training features, including the initial bias unit.
%h = the number of units in the hidden layer - NOT including the bias unit
%r = the number of output classifications 

m = size(X, 1);
n = size(X, 2) + 1; % add 1 for bias
h = hidden_layer_size;
r = num_labels;

y_matrix = eye(num_labels)(y,:); 

d3 = a3 - y_matrix;
z2;

% MAYBE OK, MAYBE NOT
d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2);


sigmoidGradient(z2);

Delta1 = d2' * a1;
Delta2 = d3' * a2;
%
%Delta1
%Delta2


Theta1_grad = Delta1/m;
Theta2_grad = Delta2/m;

%("sizes")
%
%size(a1)
%size(z2)
%size(a2)
%size(a3)
%size(d3)
%size(d2)
%size(Theta1)
%size(Delta1)
%size(Theta1_grad)
%size(Theta2)
%size(Delta2)
%size(Theta2_grad)
%
%("should be: a1: 5000x401")
%("z2: 5000x25")
%("a2: 5000x26")
%("a3: 5000x10")
%("d3: 5000x10")
%("d2: 5000x25")
%("Theta1, Delta1 and Theta1_grad: 25x401")
%("Theta2, Delta2 and Theta2_grad: 10x26")
%
%pause;

% -------------------------------------------------------------

% PART 2 - REGULARIZE GRADIENTS

Theta1(:,1) = 0;
Theta2(:,1) = 0;

%Theta1 = Theta1 * (lambda/m) + Delta1/m;
%Theta2 = Theta2 * (lambda/m) + Delta2/m;
%
%Theta1(:,1) = Delta1(:,1)/m;
%Theta2(:,1) = Delta2(:,1)/m;

%
%Theta1_grad
%Theta2_grad

Theta1_grad = Theta1_grad + Theta1*(lambda/m);
Theta2_grad = Theta2_grad + Theta2*(lambda/m);

%
%Theta1
%Theta2
%Theta1_grad
%Theta2_grad


% print vars  ALL ARE FINE FOR TEST CASE!
%d2
%d3
%Delta1
%Delta2
%z2
%sigmoidGradient(z2)
%a2
%a3




% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];




end
