function [ avg ] = weightedAvg(weights, values)
% DESCRIPTION:
%       Computes the weighted average of values by applying the weights
%
% INPUT:
%       values: Data points to average (one per row)
%       weights: Weight to apply to each data point (one per row)
%
% OUTPUT:
%       val: Weighted average of 'values'

    % Dot product between two vectors to apply weights to values
    avg = weights' * values;

    % Divide by the sum of the weights
    avg = avg ./ sum(weights, 1);
    
end