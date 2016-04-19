function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

%X = num_movies * num_features;
%Theta = num_users * num_features;
%Y = num_movies * num_users;
%R = num_movies * num_users;

% 4 x 5
[m, n] = size(R);

% FIND COST J %%%%%%%%%%%%%%%%%%%%%%

% for each user 
for j = 1:n

  % all of this user's reviews
  % 5 x 1
  this_R = R(:,j);
  
  % this user's features  
  % 1 x 3
  this_theta = Theta(j,:);
  
  % user's features * movie features
  % 1 x 5 = 1 x 3 * 3 x 5
  this_prod = this_theta * X';
  
%  
%  % 5 x 1 = 5 x 1 .* 5 x 1
%  this_prod = this_prod' .* this_R;
%  
%  this_prod = this_prod';
  
  % 5 x 1 = 5 x 1 - 5 x 1
  this_diff = this_prod' - Y(:,j);

  % only do it for R == 1  (reviewed)
  this_diff = this_diff .* this_R;
  
  %  1 x 1 = 1 x 5 * 5 x 1
  this_sum_squared = this_diff' * this_diff; 

  J += this_sum_squared;
   
  % REGULARIZED COST TERM

  %     1 x 1 * 1 x 3 * 3 x 1
  J += (lambda) * this_theta * this_theta';

 
   
end

% divide J by 2 at end 


% FIND COST GRADIENT %%%%%%%%%%%%%%%%%%%%%%


 % GRADIENT  (R already handled)
  %                5 x 1 * 1 x 1 ????
 
  % for each user 
for j = 1:n

  % all of this user's reviews
  % 5 x 1
  this_R = R(:,j);
  
  % this user's features  
  % 1 x 3
  this_theta = Theta(j,:);
  
  % user's features * movie features
  % 1 x 5 = 1 x 3 * 3 x 5
  this_prod = this_theta * X';
  
%  
%  % 5 x 1 = 5 x 1 .* 5 x 1
%  this_prod = this_prod' .* this_R;
%  
%  this_prod = this_prod';
  
  % 5 x 1 = 5 x 1 - 5 x 1
  this_diff = this_prod' - Y(:,j);

  % only do it for R == 1  (reviewed) for i = 1:num_users

  % 5 x 1
  this_diff = this_diff .* this_R;

%  X - num_movies  x num_features matrix of movie features
%  Theta - num_users  x num_features matrix of user features 

  %  1 x 3 = 1 x 5 * 5 x 3
  Theta_grad(j,:) = this_diff' * X; 
  
  %  1 x 3 = 1 x 1 * 1 x 3
  Theta_grad(j,:) += lambda * this_theta;
  
end

  
for i = 1:num_movies

  % all of this movie's reviews
  % 1 x 4
  this_R = R(i,:);

  % this movie's features
  % 1 x 3
  this_X = X(i,:);
  
  % user's features * movie features
  % 4 x 1 = 4 x 3 * 3 x 1
  this_prod = Theta * this_X';

  % 1 x 4 = 1 x 4 - 1 x 4
  this_diff = this_prod' - Y(i,:);

  % only do it for R == 1  (reviewed) for i = 1:num_users
  %  1 x 4 =   1 x 4 .* 1 x 4
  this_diff = this_diff .* this_R;
  
  % 1 x 3 = 1 x 4 * 4 x 3
  X_grad(i,:) = this_diff * Theta;
  
  % 1 x 3 = 1 x 1 * 1 x 3
  X_grad(i,:) += lambda * this_X;
  
  %     1 x 1 * 1 x 3 * 3 x 1
  J += (lambda) * this_X * this_X';
  
end


J /= 2;


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
