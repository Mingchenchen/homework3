clear
load('diabetes.mat');

% Add bias coefficients
x_train = [ones(size(x_train, 1), 1), x_train];
x_test = [ones(size(x_test, 1), 1), x_test];

lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10];
test_error = zeros(size(lambdas));
train_error = zeros(size(lambdas));
for i = 1:size(lambdas, 2)
	%w = ridge_regression(x_train, y_train, lambdas(i));
    w = ridge(y_train, x_train, lambdas(i), 1);
    train_error(i) = mean((y_train - x_train * w).^2);
    test_error(i) = mean((y_test - x_test * w).^2);
end

loglog(lambdas, train_error, 'r', lambdas, test_error, 'b');
title('{\bf Training and Testing Error vs. lambda}')
xlabel('lambda (1e-5 to 10)')
ylabel('Mean Squared Error')
legend('Training Error', 'Testing Error')