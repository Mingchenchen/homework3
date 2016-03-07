clear
load('diabetes.mat');

% Add bias coefficients
x_train = [ones(size(x_train, 1), 1), x_train];
x_test = [ones(size(x_test, 1), 1), x_test];

lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10];
test_error = zeros(size(lambdas));
train_error = zeros(size(lambdas));

% Train models using log-scale lambda values
for i = 1:size(lambdas, 2)
	w = ridge_regression(x_train, y_train, lambdas(i));
    train_error(i) = mean((y_train - x_train * w).^2);
    test_error(i) = mean((y_test - x_test * w).^2);
end

loglog(lambdas, train_error, 'r', lambdas, test_error, 'b');
hold on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do 5-fold cross-validation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Partition the data
folds = cvpartition(y_train, 'KFold', 5);
cv_error = zeros(size(lambdas, 2), 1);
for i = 1:size(lambdas, 2)
    lambda_cv_error = zeros(5, 1);
    for j = 1:5
        train_idx = folds.training(j);
        test_idx = folds.test(j);
        w = ridge_regression(x_train(train_idx, :), y_train(train_idx),...
            lambdas(i));
        lambda_cv_error(j) = mean((y_train(test_idx) - x_train(test_idx, :) * w).^2);
    end
    cv_error(i) = mean(lambda_cv_error);
end

[best_error, best_idx] = min(cv_error);

scatter(lambdas(best_idx), best_error, 'xg');
title('{\bf Training and Testing Error vs. lambda}')
xlabel('lambda (1e-5 to 10)')
ylabel('Mean Squared Error')
legend('Training Error', 'Testing Error', 'Best CV lambda', 'Location',...
    'northwest')