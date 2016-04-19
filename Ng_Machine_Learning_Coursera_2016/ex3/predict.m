function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Theta1 and Theta2 contain the parameters for each unit in rows.
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26

% add ones for bias -> a1 = [5000 x 401]
a1 = [ones(m, 1) X];


% a1 = [5000 x 401], Theta1 has size 25 x 401
% z2 = 5000 x 25
z2 = sigmoid(a1 * Theta1');

% a2 = 5000 x 26  (add one for bias)
a2 = [ones(m, 1) z2];

% a2 = 5000 x 26 Theta2 has size 10 x 26
% a3 = [5000 x 10]
a3 = sigmoid(a2 * Theta2');

% get a vector max values of a3 for each row -> the predictions for each data point
[my_max, index] = max(a3, [], 2);

p = index

% =========================================================================


end
