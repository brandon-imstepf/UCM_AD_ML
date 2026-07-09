function output = apply_sr_equation(input_values, equation_func)
    % Load scaling parameters
    load('scaling_params.mat');
    
    % Scale inputs
    scaled_inputs = (input_values - feature_means) ./ feature_stds;
    
    % Apply equation (in scaled space)
    scaled_output = equation_func(scaled_inputs);
    
    % Unscale output
    output = scaled_output * target_std + target_mean;
end
