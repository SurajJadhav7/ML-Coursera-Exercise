function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha
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
    a=theta(1)-alpha*(1/m)*sum(X*theta-y);
    b=theta(2)-alpha*(1/m)*sum((X*theta-y).*X(:,2));
    theta(1)=a
    theta(2)=b
    % ============================================================
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end
end
