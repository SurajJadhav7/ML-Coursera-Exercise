function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
m = length(y);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    temp=zeros(size(X,2),1);
    for i = 1:size(X,2)
        temp(i)=theta(i)-alpha*(1/m)*sum((X*theta-y).*X(:,i));
    end
    for i = 1:size(X,2)
        theta(i)=temp(i);
    end
    J_history(iter) = computeCostMulti(X, y, theta);
end
end
