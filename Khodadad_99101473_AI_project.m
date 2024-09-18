%% AI & Computational Intelligence Projet
% Nommy Khodadad
% 99101473
%% Part 1 ( Feature Extraction ):
%Loading the data
clear all; 
clc; 
close all;
data = load('Project_data.mat');
test_data = data.TestData(: , 1001:end, :); %Removing the first second
train_data = data.TrainData(: , 1001:end, :); %Removing the first second
train_label = data.TrainLabels;
fs = 1000;
[Channels_num, Samples_num, Trials_num] = size(train_data);
N_positive = length(find(train_label == 1));
N_negative = length(find(train_label == -1));
%% In this sub-part1, we are calculating the Fisher scores for 4 different standards of the channels:
clc;
for channel = [1: 1: Channels_num]   
    %Loading the data for each channel
    TD = train_data(channel, :, :);
    TD = transpose(squeeze(TD));
    %Calculating the mean , variance , medium frequency ,and mean frequency for each trial
    for i = [1: 1: Trials_num]
        average(channel, i) = sum(TD(i, :)) / Samples_num;
        variance(channel, i) = var(TD(i, :));
        Medium_Frequency(channel, i) = medfreq(TD(i, :));
        Mean_Frequency(channel, i) = meanfreq(TD(i, :));
    end
    %Calculating Fisher scores from some standards for each feature:
    %Here we determine the Fisher score for channel mean , variance ,
    %medium frequency , and mean frequency
    label_pos = find(train_label == 1);
    label_neg = find(train_label == -1);
    u0_mean = sum( average(channel, :) ) / length(average);
    u1_mean = sum( average(channel, label_pos) ) / length(label_pos);
    u2_mean = sum( average(channel, label_neg) ) / length(label_neg);
    v1_mean = var( average(channel, label_pos) );
    v2_mean = var( average(channel, label_neg) );
    u0_variance = sum( variance(channel, :) ) / length(variance);
    u1_variance = sum( variance(channel, label_pos) ) / length(label_pos);
    u2_variance = sum( variance(channel, label_neg) ) / length(label_neg);
    v1_variance = var( variance(channel, label_pos) );
    v2_variance = var( variance(channel, label_neg) );
    u0_MedFreq = sum( Medium_Frequency(channel, :) ) / length(Medium_Frequency);
    u1_MedFreq = sum( Medium_Frequency(channel, label_pos) ) / length(label_pos);
    u2_MedFreq = sum( Medium_Frequency(channel, label_neg) ) / length(label_neg);
    v1_MedFreq = var( Medium_Frequency(channel, label_pos) );
    v2_MedFreq = var( Medium_Frequency(channel, label_neg) );
    u0_MeanFreq = sum( Mean_Frequency(channel, :) ) / length(Mean_Frequency);
    u1_MeanFreq = sum( Mean_Frequency(channel, label_pos) ) / length(label_pos);
    u2_MeanFreq = sum( Mean_Frequency(channel, label_neg) ) / length(label_neg);
    v1_MeanFreq = var( Mean_Frequency(channel, label_pos) );
    v2_MeanFreq = var( Mean_Frequency(channel, label_neg) );
    J_mean(channel) = ((u0_mean - u1_mean)^2 + (u0_mean - u2_mean)^2) / (v1_mean + v2_mean); 
    J_variance(channel) = ((u0_variance - u1_variance)^2 + (u0_variance - u2_variance)^2) / (v1_variance + v2_variance); 
    J_MedFreq(channel) = ((u0_MedFreq - u1_MedFreq)^2 + (u0_MedFreq - u2_MedFreq)^2) / (v1_MedFreq + v2_MedFreq);
    J_MeanFreq(channel) = ((u0_MeanFreq - u1_MeanFreq)^2 + (u0_MeanFreq - u2_MeanFreq)^2) / (v1_MeanFreq + v2_MeanFreq);
end
%Normalizing the feature matrices
average = normalize(average);
variance = normalize(variance);
Medium_Frequency = normalize(Medium_Frequency);
Mean_Frequency = normalize(Mean_Frequency);
%Displaying the best channel(feature) for each property(standard) mentioned
%above:
disp(['Best mean Fisher score is ', num2str(max(J_mean)), ' from channel = ', num2str(find(J_mean == max(J_mean)))]);
disp(['Best variance Fisher score is ', num2str(max(J_variance)), ' from channel = ', num2str(find(J_variance == max(J_variance)))]);
disp(['Best medium frequency Fisher score is ', num2str(max(J_MedFreq)), ' from channel = ', num2str(find(J_MedFreq == max(J_MedFreq)))]);
disp(['Best mean frequency Fisher score is ', num2str(max(J_MeanFreq)), ' from channel = ', num2str(find(J_MeanFreq == max(J_MeanFreq)))]);
J_matrix = [J_mean ; J_variance ; J_MedFreq ; J_MeanFreq];
%% In this sub-part2, we are calculating the Fisher scores for another 4 different standards of the channels:
%Note that I had to devide these 2 sub-parts so that the process time would
%reach minimum.
%clc;
for channel = [1: 1: Channels_num]   
    %Loading the data for each channel
    TD = train_data(channel, :, :);
    TD = transpose(squeeze(TD));
    %Calculating the Entropy , Skewness , Band Power , and Bandwidth for each trial
    for i = [1: 1: Trials_num]
        Entropy(channel, i) = entropy(TD(i, :));
        Skewness(channel, i) = skewness(TD(i, :));
        Band_Power(channel, i) = bandpower(TD(i, :));
        Bandwidth(channel, i) = obw(TD(i, :));
    end
    %Calculating Fisher scores from some standards for each feature:
    %Here we determine the Fisher score for channel Entropy , Skewness ,
    %Band Power , and 99% Bandwidth
    label_pos = find(train_label == 1);
    label_neg = find(train_label == -1);
    u0_entropy = sum( Entropy(channel, :) ) / length(Entropy);
    u1_entropy = sum( Entropy(channel, label_pos) ) / length(label_pos);
    u2_entropy = sum( Entropy(channel, label_neg) ) / length(label_neg);
    v1_entropy = var( Entropy(channel, label_pos) );
    v2_entropy = var( Entropy(channel, label_neg) );
    u0_skewness = sum( Skewness(channel, :) ) / length(Skewness);
    u1_skewness = sum( Skewness(channel, label_pos) ) / length(label_pos);
    u2_skewness = sum( Skewness(channel, label_neg) ) / length(label_neg);
    v1_skewness = var( Skewness(channel, label_pos) );
    v2_skewness = var( Skewness(channel, label_neg) );
    u0_Bandpower = sum( Band_Power(channel, :) ) / length(Band_Power);
    u1_Bandpower = sum( Band_Power(channel, label_pos) ) / length(label_pos);
    u2_Bandpower = sum( Band_Power(channel, label_neg) ) / length(label_neg);
    v1_Bandpower = var( Band_Power(channel, label_pos) );
    v2_Bandpower = var( Band_Power(channel, label_neg) );
    u0_Bandwidth = sum( Bandwidth(channel, :) ) / length(Bandwidth);
    u1_Bandwidth = sum( Bandwidth(channel, label_pos) ) / length(label_pos);
    u2_Bandwidth = sum( Bandwidth(channel, label_neg) ) / length(label_neg);
    v1_Bandwidth = var( Bandwidth(channel, label_pos) );
    v2_Bandwidth = var( Bandwidth(channel, label_neg) );
    J_entropy(channel) = ((u0_entropy - u1_entropy)^2 + (u0_entropy - u2_entropy)^2) / (v1_entropy + v2_entropy); 
    J_skewness(channel) = ((u0_skewness - u1_skewness)^2 + (u0_skewness - u2_skewness)^2) / (v1_skewness + v2_skewness); 
    J_Bandpower(channel) = ((u0_Bandpower - u1_Bandpower)^2 + (u0_Bandpower - u2_Bandpower)^2) / (v1_Bandpower + v2_Bandpower);
    J_Bandwidth(channel) = ((u0_Bandwidth - u1_Bandwidth)^2 + (u0_Bandwidth - u2_Bandwidth)^2) / (v1_Bandwidth + v2_Bandwidth);
end
%Normalizing the feature matrices
Entropy = normalize(Entropy);
Skewness = normalize(Skewness);
Band_Power = normalize(Band_Power);
Bandwidth = normalize(Bandwidth);
%Displaying the best channel(feature) for each property(standard) mentioned
%above:
disp(['Best Entropy Fisher score is ', num2str(max(J_entropy)), ' from channel = ', num2str(find(J_entropy == max(J_entropy)))]);
disp(['Best Skewness Fisher score is ', num2str(max(J_skewness)), ' from channel = ', num2str(find(J_skewness == max(J_skewness)))]);
disp(['Best Band Power Fisher score is ', num2str(max(J_Bandpower)), ' from channel = ', num2str(find(J_Bandpower == max(J_Bandpower)))]);
disp(['Best Bandwidth Fisher score is ', num2str(max(J_Bandwidth)), ' from channel = ', num2str(find(J_Bandwidth == max(J_Bandwidth)))]);
J_matrix = [J_matrix ; J_entropy ; J_skewness ; J_Bandpower ; J_Bandwidth];
%% Maximum Power Frequency
clc;
for channel = [1: 1: Channels_num]
    TD = train_data(channel, :, :);
    TD = transpose(squeeze(TD));
    for i = [1: 1: Trials_num]
        x = TD(i, :);
        n = length(x);
        y = fftshift(fft(x));
        f = (-n/2:n/2-1)*(fs/n);
        power = abs(y).^2/n;       
        index = find(power == max(power));
        Max_Power_Frequency(channel, i) = index(end);
    end
    %Fisher Score for Feature Selection
    label_pos = find(train_label == 1);
    label_neg = find(train_label == -1);
    u0 = sum(Max_Power_Frequency(channel, :)) / length(Max_Power_Frequency);
    u1 = sum(Max_Power_Frequency(channel, label_pos)) / length(label_pos);
    u2 = sum(Max_Power_Frequency(channel, label_neg)) / length(label_neg);
    var1 = var(Max_Power_Frequency(channel, label_pos));
    var2 = var(Max_Power_Frequency(channel, label_neg));
    J_max_power_freq(channel) = ((u0 - u1)^2 + (u0 - u2)^2) / (var1 + var2); 
end
Max_Power_Frequency = normalize(Max_Power_Frequency);
disp(['Best Maximum Power Frequency Fisher score is ', num2str(max(J_max_power_freq)), ' for channel = ', num2str(find(J_max_power_freq == max(J_max_power_freq)))]);
J_matrix = [J_matrix ; J_max_power_freq];
%%
clc;
%Calculate correlations between channels using corr function
corr_matrix = corr(reshape(train_data, [], Channels_num));
%Exclude correlation of a channel with itself
corr_matrix(logical(eye(size(corr_matrix)))) = 0;
%Find the pair of channels with the highest correlation
%[max_corr, max_corr_indices] = max(corr_matrix(:));
%[channel1, channel2] = ind2sub(size(corr_matrix), max_corr_indices);
%disp(['Max correlation is ', num2str(max_corr), ' between channel ', num2str(channel1), ' and channel ', num2str(channel2)]);
% Find the top 10 covariance pairs excluding self-correlation
num_pairs = 10;
best_corr_indices = [];
for k = 1:num_pairs
    [max_corr, max_corr_index] = max(abs(corr_matrix(:)));
    [channel1, channel2] = ind2sub(size(corr_matrix), max_corr_index);
    best_corr_indices = [best_corr_indices; channel1, channel2, max_corr];
    corr_matrix(max_corr_index) = NaN; % Exclude this correlation from further consideration
end
disp('Top 10 covariance pairs:');
disp(best_corr_indices);
%Best covariances are between channels:
%11 & 23 , 19 & 25 , 50 & 44 , 30 & 39 , 25 & 37
%After cutting the first second:
% 1 & 23 , 53 & 38 , 54 & 17 , 15 & 56 , 19 & 4
%%
clc;
%Calculate correlations between channels in J_matrix using corr function
corr_matrix = corr(reshape(J_matrix, [], Channels_num));
%Exclude correlation of a channel with itself
corr_matrix(logical(eye(size(corr_matrix)))) = 0;
%Find the pair of channels with the highest correlation
%[max_corr, max_corr_indices] = max(corr_matrix(:));
%[channel1, channel2] = ind2sub(size(corr_matrix), max_corr_indices);
%disp(['Max correlation is ', num2str(max_corr), ' between channel ', num2str(channel1), ' and channel ', num2str(channel2)]);
num_pairs = 10;
best_corr_indices = [];
for k = 1:num_pairs
    [max_corr, max_corr_index] = max(abs(corr_matrix(:)));
    [channel1, channel2] = ind2sub(size(corr_matrix), max_corr_index);
    best_corr_indices = [best_corr_indices; channel1, channel2, max_corr];
    corr_matrix(max_corr_index) = NaN; % Exclude this correlation from further consideration
end
disp('Top 10 covariance pairs:');
disp(best_corr_indices);
%Best covariances are between channels:
%59 & 56, 24 & 30 , 25 & 9 , 11 & 25 , 11 & 9 ==> 9,11,25 + 24,30 + 56,59
%After cutting the first second:
% 28 & 27, 11 & 9 , 36 & 12 , 5 & 7 , 23 & 17
%% feature selection sub-part 3:
clc;
%After running the previous section, these were the outputs:
%Best mean Fisher score is 0.019364 from channel = 47
%Best variance Fisher score is 0.015599 from channel = 43
%Best medium frequency Fisher score is 0.030914 from channel = 43
%Best mean frequency Fisher score is 0.02443 from channel = 23
%Best Entropy Fisher score is 0.01804 from channel = 50
%Best Skewness Fisher score is 0.021747 from channel = 11
%Best Band Power Fisher score is 0.015561 from channel = 43
%Best Bandwidth Fisher score is 0.016727 from channel = 41
%Best Maximum Power Frequency Fisher score is 0.023661 for channel = 24
%Since we have 59 channels, let's manually select the 2nd best channel for
%each standard from the J_matrix: channels 33 , 25 , 56 , and 43 again!!
%After deleting the first second:
%Best mean Fisher score is 0.015768 from channel = 23
%Best variance Fisher score is 0.016018 from channel = 43
%Best medium frequency Fisher score is 0.026322 from channel = 43
%Best mean frequency Fisher score is 0.023617 from channel = 23
%Best Entropy Fisher score is 0.021742 from channel = 50
%Best Skewness Fisher score is 0.024515 from channel = 25
%Best Band Power Fisher score is 0.015967 from channel = 43
%Best Bandwidth Fisher score is 0.016115 from channel = 14
%also these channels are good together:
% 1 & 23 , 53 & 38 , 54 & 17 , 15 & 56 , 19 & 4
% 28 & 27, 11 & 9 , 36 & 12 , 5 & 7 , 23 & 17
Final_Features_Mat = transpose([average(1, :);average(23, :);average(17, :);average(54, :);average(56, :); variance(43, :); ...
    average(6, :);average(28, :);variance(53, :);average(38, :);variance(56, :);variance(50, :);variance(15, :);variance(13, :); ...
    variance(6, :);Medium_Frequency(56, :);Medium_Frequency(54, :);Medium_Frequency(59, :); ...
    Medium_Frequency(33, :);Mean_Frequency(23, :);Medium_Frequency(43, :);Medium_Frequency(17, :); ...
    Medium_Frequency(23, :);Medium_Frequency(26, :);Mean_Frequency(43, :);Medium_Frequency(1, :); ...
    Mean_Frequency(11, :);Mean_Frequency(15, :);Mean_Frequency(56, :);Mean_Frequency(31, :);Mean_Frequency(54, :); ...
    Mean_Frequency(59, :);Entropy(43, :);Entropy(35, :);Skewness(25, :);Skewness(41, :);Skewness(49, :); ...
    Entropy(50, :);Entropy(52, :);Entropy(15, :); Skewness(11, :);Skewness(9, :);Skewness(25, :);Max_Power_Frequency(54, :); ...
    Skewness(38, :);Skewness(32, :); Bandwidth(41, :);Bandwidth(25, :);Bandwidth(14, :);Skewness(50, :);Max_Power_Frequency(1, :); ...
    Mean_Frequency(33, :); Max_Power_Frequency(24, :);Max_Power_Frequency(43, :);Max_Power_Frequency(23, :);Max_Power_Frequency(17, :); ...
    Max_Power_Frequency(51, :); Band_Power(43, :);Band_Power(50, :);Band_Power(13, :);Band_Power(6, :);Band_Power(15, :)]);
%Features For Test Data
[Channels_num_Test, Samples_num_Test, Trials_num_Test] = size(test_data);
for channel = [1: 1: Channels_num_Test]    
    %Loading the data for each channel in test dataset
    TD = test_data(channel, :, :);
    TD = transpose(squeeze(TD));    
    % Calculating variance for each trial
    for i = [1: 1: Trials_num_Test]
        average_test(channel, i) = sum(TD(i, :)) / Samples_num_Test;
        variance_test(channel, i) = var(TD(i, :));
        Medium_Frequency_test(channel, i) = medfreq(TD(i, :));
        Mean_Frequency_test(channel, i) = meanfreq(TD(i, :));
        Skewness_test(channel, i) = skewness(TD(i, :));
        Entropy_test(channel, i) = entropy(TD(i, :));
        Band_Power_test(channel, i) = bandpower(TD(i, :));
        Bandwidth_test(channel, i) = obw(TD(i, :));
        xT = TD(i, :);
        nT = length(xT);
        yT = fftshift(fft(xT));
        fT = (-nT/2:nT/2-1)*(fs/n);     
        powerT = abs(yT).^2/nT;           
        indexT = find(powerT == max(powerT));
        Max_Power_Frequency_test(channel, i) = indexT(end);
    end
end
average_test = normalize(average_test);
variance_test = normalize(variance_test);
Medium_Frequency_test = normalize(Medium_Frequency_test);
Mean_Frequency_test = normalize(Mean_Frequency_test);
Skewness_test = normalize(Skewness_test);
Entropy_test = normalize(Entropy_test);
Band_Power_test = normalize(Band_Power_test);
Bandwidth_test = normalize(Bandwidth_test);
Max_Power_Frequency_test = normalize(Max_Power_Frequency_test);
Final_Test_Features_Mat = transpose([average_test(1, :);average_test(23, :);average_test(17, :);average_test(54, :);average_test(56, :); variance_test(43, :); ...
    average_test(6, :);average_test(28, :);variance_test(53, :);average_test(38, :);variance_test(56, :);variance_test(50, :);variance_test(15, :);variance_test(13, :); ...
    variance_test(6, :);Medium_Frequency_test(56, :);Medium_Frequency_test(54, :);Medium_Frequency_test(59, :); ...
    Medium_Frequency_test(33, :);Mean_Frequency_test(23, :);Medium_Frequency_test(43, :);Medium_Frequency_test(17, :); ...
    Medium_Frequency_test(23, :);Medium_Frequency_test(26, :);Mean_Frequency_test(43, :);Medium_Frequency_test(1, :); ...
    Mean_Frequency_test(11, :);Mean_Frequency_test(15, :);Mean_Frequency_test(56, :);Mean_Frequency_test(31, :);Mean_Frequency_test(54, :); ...
    Mean_Frequency_test(59, :);Entropy_test(43, :);Entropy_test(35, :);Skewness_test(25, :);Skewness_test(41, :);Skewness_test(49, :); ...
    Entropy_test(50, :);Entropy_test(52, :);Entropy_test(15, :); Skewness_test(11, :);Skewness_test(9, :);Skewness_test(25, :);Max_Power_Frequency_test(54, :); ...
    Skewness_test(38, :);Skewness_test(32, :); Bandwidth_test(41, :);Bandwidth_test(25, :);Bandwidth_test(14, :);Skewness_test(50, :);Max_Power_Frequency_test(1, :); ...
    Mean_Frequency_test(33, :); Max_Power_Frequency_test(24, :);Max_Power_Frequency_test(43, :);Max_Power_Frequency_test(23, :);Max_Power_Frequency_test(17, :); ...
    Max_Power_Frequency_test(51, :); Band_Power_test(43, :);Band_Power_test(50, :);Band_Power_test(13, :);Band_Power_test(6, :);Band_Power_test(15, :)]);
%% Part 2 (After choosing 62 features, we start the MLP algorithm to determine the remaining 159 test labels and saving them):
clc;
%Defining activation functions
activation_functions = ["satlin","logsig","radbas","hardlims","purelin","tansig"];
%Initializing best accuracy and the corresponding parameters
best_accuracy = 0;
best_N = zeros(1, length(activation_functions));
best_activation_function = cell(1, length(activation_functions));
Final_Features_Mat = transpose(Final_Features_Mat);
%Loop over each activation function
for i = 1:length(activation_functions)
    %Loop over a range of neurons in the hidden layer
    for N = 1:20
        %Storing accuracies for each fold initially giving the zero value
        accuracies = zeros(1, 5); 
        %Performing 5-fold cross-validation
        for k = 1:5
            %Split data into training and validation sets
            train_indices = [1:(k-1)*110, k*110+1:550];
            valid_indices = (k-1)*110+1:k*110;
            TrainX = Final_Features_Mat(:, train_indices);
            ValX = Final_Features_Mat(:, valid_indices);
            TrainY = train_label(train_indices);
            ValY = train_label(valid_indices);
            %Creating and training MLP
            net = patternnet(N);
            net = train(net, TrainX, TrainY);
            %Setting activation function
            net.layers{2}.transferFcn = activation_functions{i};
            %Make predictions on validation set
            predict_y = net(ValX);
            % Apply threshold to separate classes
            predict_y(predict_y < 0) = -1;
            predict_y(predict_y >= 0) = 1;
            %Calculate accuracy
            accuracies(k) = sum(predict_y == ValY) / length(ValY);
            if accuracies(k) > best_accuracy
                best_accuracy = accuracies(k);
                best_N(i) = N;
                best_activation_function{i} = activation_functions{i};
            end
        end
        
        %Calculating the average accuracy over all folds
        avg_accuracy = mean(accuracies);
        %Updating best accuracy and parameters if current model is better
        if avg_accuracy > best_accuracy
            best_accuracy = avg_accuracy;
            best_N(i) = N;
            best_activation_function{i} = activation_functions{i};
        end
    end
    % Display results for each activation function
    disp(['Activation function: ', activation_functions{i}, ', Best neuron quantity: ', num2str(best_N(i)), ', Accuracy = ', num2str(best_accuracy)]);
    best_accuracy = 0;
end
Final_Features_Mat = transpose(Final_Features_Mat);
%% training the best MLP network:
clc;
%Defining activation functions
activation_functions = ["satlin","purelin","tansig"];
%Initializing best accuracy and the corresponding parameters
best_accuracy = 0;
best_N = zeros(1, length(activation_functions));
best_activation_function = cell(1, length(activation_functions));
Final_Features_Mat = transpose(Final_Features_Mat);
%Loop over each activation function
for i = 1:length(activation_functions)
    %Loop over a range of neurons in the hidden layer
    for N = 10:20
        %Storing accuracies for each fold initially giving the zero value
        accuracies = zeros(1, 5); 
        %Performing 5-fold cross-validation
        for k = 1:5
            %Split data into training and validation sets
            train_indices = [1:(k-1)*110, k*110+1:550];
            valid_indices = (k-1)*110+1:k*110;
            TrainX = Final_Features_Mat(:, train_indices);
            ValX = Final_Features_Mat(:, valid_indices);
            TrainY = train_label(train_indices);
            ValY = train_label(valid_indices);
            %Creating and training MLP
            net = patternnet(N);
            net = train(net, TrainX, TrainY);
            %Setting activation function
            net.layers{2}.transferFcn = activation_functions{i};
            %Make predictions on validation set
            predict_y = net(ValX);
            % Apply threshold to separate classes
            predict_y(predict_y < 0) = -1;
            predict_y(predict_y >= 0) = 1;
            %Calculate accuracy
            accuracies(k) = sum(predict_y == ValY) / length(ValY);
            if accuracies(k) > best_accuracy
                best_accuracy = accuracies(k);
                best_N(i) = N;
                best_activation_function{i} = activation_functions{i};
            end
        end
        
        %Calculating the average accuracy over all folds
        avg_accuracy = mean(accuracies);
        %Updating best accuracy and parameters if current model is better
        if avg_accuracy > best_accuracy
            best_accuracy = avg_accuracy;
            best_N(i) = N;
            best_activation_function{i} = activation_functions{i};
        end
    end
    % Display results for each activation function
    disp(['Activation function: ', activation_functions{i}, ', Best neuron quantity: ', num2str(best_N(i)), ', Accuracy = ', num2str(best_accuracy)]);
    best_accuracy = 0;
end
Final_Features_Mat = transpose(Final_Features_Mat);
%% Predicting MLP labels for the test data using the best MLP network
clc;
Final_Test_Features_Mat = transpose(Final_Test_Features_Mat);
predict_y_test = sim(net, Final_Test_Features_Mat);
predict_y_test(predict_y_test <= 0) = -1;
predict_y_test(predict_y_test > 0) = 1;
%Saving the predicted labels to a .mat file
save('predicted_MLP_labels.mat', 'predict_y_test');
%% Part 3 (Using RBF algorithm instead of MLP)
clc;
% Defining activation functions
activation_functions = ["radbas"];
% Initializing best accuracy and the corresponding parameters
best_accuracy = 0;
best_N = zeros(1, length(activation_functions));
best_activation_function = cell(1, length(activation_functions));
Final_Features_Mat = transpose(Final_Features_Mat);
% Storing accuracies for each fold initially giving the zero value
accuracies = zeros(1, 5); 
% Performing 5-fold cross-validation
for k = 1:5
    % Split data into training and validation sets
    train_indices = [1:(k-1)*110, k*110+1:550];
    valid_indices = (k-1)*110+1:k*110;
    TrainX = Final_Features_Mat(:, train_indices);
    ValX = Final_Features_Mat(:, valid_indices);
    TrainY = train_label(train_indices);
    ValY = train_label(valid_indices);
    %Creating and training RBF network
    net = newrb(TrainX, TrainY, 0, 10);          
    %Make predictions on validation set
    predict_y = sim(net, ValX);
    % Apply threshold to separate classes
    predict_y(predict_y < 0) = -1;
    predict_y(predict_y >= 0) = 1;
    % Calculate accuracy
    accuracies(k) = sum(predict_y == ValY) / length(ValY);
end        
%Calculating the average accuracy over all folds
avg_accuracy = mean(accuracies);
%Display results for each activation function
disp(['Activation function: ', activation_functions{i}, ', Accuracy = ', num2str(avg_accuracy)]);
Final_Features_Mat = transpose(Final_Features_Mat);
%% Predicting RBF labels for the test data using the RBF network
clc;
Final_Test_Features_Mat = transpose(Final_Test_Features_Mat);
predict_y_test = sim(net, Final_Test_Features_Mat);
predict_y_test(predict_y_test <= 0) = -1;
predict_y_test(predict_y_test > 0) = 1;
%Saving the predicted labels to a .mat file
save('predicted_RBF_labels.mat', 'predict_y_test');
%% Phase 2 (Evolutionary Algorithms):
clc;
%Selecting the top 60 features from the Fisher matrix: J_matrix
J_mat_reshaped = reshape(J_matrix, [1, 531]);
J_vector_sorted = sort(J_mat_reshaped);
top_60_FisherVals = J_vector_sorted(end - 59:end);
%finding the corresponding features and values
for i = [1: 1: 60]
    [feature(i), channel(i)] = find(J_matrix == top_60_FisherVals(i));
end
for i = [1: 1: 60]
    if (feature(i) == 1)
        BestFeatureMat_GA(i, :) = average(channel(i), :); 
    end
    if (feature(i) == 2)
        BestFeatureMat_GA(i, :) = variance(channel(i), :); 
    end
    if (feature(i) == 6)
        BestFeatureMat_GA(i, :) = Skewness(channel(i), :); 
    end    
    if (feature(i) == 5)
        BestFeatureMat_GA(i, :) = Entropy(channel(i), :); 
    end    
    if (feature(i) == 3)
        BestFeatureMat_GA(i, :) = Medium_Frequency(channel(i), :); 
    end    
    if (feature(i) == 4)
        BestFeatureMat_GA(i, :) = Mean_Frequency(channel(i), :); 
    end
    if (feature(i) == 7)
        BestFeatureMat_GA(i, :) = Band_Power(channel(i), :); 
    end   
    if (feature(i) == 8)
        BestFeatureMat_GA(i, :) = Bandwidth(channel(i), :); 
    end
    if (feature(i) == 9)
        BestFeatureMat_GA(i, :) = Max_Power_Frequency(channel(i), :); 
    end  
end
%%
clc;
%Defining the Evolutionary Algorithm parameters
population_size = 100;
num_generations = 50;
crossover_rate = 0.8;
mutation_rate = 0.01;
num_parents = 20;
%Initializing the population randomly:
population = randi([0, 1], population_size, 60);
%Evaluating the fitness of the initial population
fitness_values = evaluate_fitness(population,BestFeatureMat_GA,train_label);
for generation = 1:num_generations
    %Selecting the parents
    parents = selection(population, fitness_values, num_parents);
    %Performing crossover
    offspring = crossover(parents, crossover_rate);
    %Performing mutation
    offspring = mutation(offspring, mutation_rate);
    %Evaluating the fitness of offspring
    offspring_fitness = evaluate_fitness(offspring,BestFeatureMat_GA ,train_label);
    %Combining parents and offspring
    combined_population = [population; offspring];
    combined_fitness = [fitness_values; offspring_fitness];
    %Sorting combined population based on fitness
    [~, sorted_indices] = sort(combined_fitness, 'descend');
    population = combined_population(sorted_indices(1:population_size), :);
    fitness_values = combined_fitness(sorted_indices(1:population_size));
    %Displaying the best fitness value in each generation
    best_fitness = max(fitness_values);
    disp(['Generation ', num2str(generation), ': Best Fitness = ', num2str(best_fitness)]);
    best_fitness = 0;
end
%Finding the best individual in the final population
best_individual_index = find(fitness_values == max(fitness_values), 1);
best_individual = population(best_individual_index, :);
%Saving the best feature set in a matrix
BestFeatureSet = best_individual;
%% Now that we know the best feature set of the 60 features, we only pick those features for training and testing:
%finding the corresponding features and values
for i = [1: 1: 60]
    [feature(i), channel(i)] = find(J_matrix == top_60_FisherVals(i));
end
for i = [1: 1: 60]
    if (feature(i) == 1)
        BestFeatureMat_GA_test(i, :) = average_test(channel(i), :); 
    end
    if (feature(i) == 2)
        BestFeatureMat_GA_test(i, :) = variance_test(channel(i), :); 
    end
    if (feature(i) == 6)
        BestFeatureMat_GA_test(i, :) = Skewness_test(channel(i), :); 
    end    
    if (feature(i) == 5)
        BestFeatureMat_GA_test(i, :) = Entropy_test(channel(i), :); 
    end    
    if (feature(i) == 3)
        BestFeatureMat_GA_test(i, :) = Medium_Frequency_test(channel(i), :); 
    end    
    if (feature(i) == 4)
        BestFeatureMat_GA_test(i, :) = Mean_Frequency_test(channel(i), :); 
    end
    if (feature(i) == 7)
        BestFeatureMat_GA_test(i, :) = Band_Power_test(channel(i), :); 
    end   
    if (feature(i) == 8)
        BestFeatureMat_GA_test(i, :) = Bandwidth_test(channel(i), :); 
    end
    if (feature(i) == 9)
        BestFeatureMat_GA_test(i, :) = Max_Power_Frequency_test(channel(i), :); 
    end  
end
%% MLP for the best GA features:
clc;
%Defining activation functions
activation_functions = ["satlin","purelin","tansig"];
%Initializing best accuracy and the corresponding parameters
best_accuracy = 0;
best_N = zeros(1, length(activation_functions));
best_activation_function = cell(1, length(activation_functions));
%BestFeatureMat_GA = transpose(BestFeatureMat_GA);
%Loop over each activation function
for i = 1:length(activation_functions)
    %Loop over a range of neurons in the hidden layer
    for N = 10:20
        %Storing accuracies for each fold initially giving the zero value
        accuracies = zeros(1, 5); 
        %Performing 5-fold cross-validation
        for k = 1:5
            %Split data into training and validation sets
            train_indices = [1:(k-1)*110, k*110+1:550];
            valid_indices = (k-1)*110+1:k*110;
            TrainX = BestFeatureMat_GA(:, train_indices);
            ValX = BestFeatureMat_GA(:, valid_indices);
            TrainY = train_label(train_indices);
            ValY = train_label(valid_indices);
            %Creating and training MLP
            net = patternnet(N);
            net = train(net, TrainX, TrainY);
            %Setting activation function
            net.layers{2}.transferFcn = activation_functions{i};
            %Make predictions on validation set
            predict_y = net(ValX);
            % Apply threshold to separate classes
            predict_y(predict_y < 0) = -1;
            predict_y(predict_y >= 0) = 1;
            %Calculate accuracy
            accuracies(k) = sum(predict_y == ValY) / length(ValY);
            if accuracies(k) > best_accuracy
                best_accuracy = accuracies(k);
                best_N(i) = N;
                best_activation_function{i} = activation_functions{i};
            end
        end
        
        %Calculating the average accuracy over all folds
        avg_accuracy = mean(accuracies);
        %Updating best accuracy and parameters if current model is better
        if avg_accuracy > best_accuracy
            best_accuracy = avg_accuracy;
            best_N(i) = N;
            best_activation_function{i} = activation_functions{i};
        end
    end
    % Display results for each activation function
    disp(['Activation function: ', activation_functions{i}, ', Best neuron quantity: ', num2str(best_N(i)), ', Accuracy = ', num2str(best_accuracy)]);
    best_accuracy = 0;
end
%Predicting MLP labels for the test data using the best MLP network 
predict_y_test = sim(net, BestFeatureMat_GA_test);
predict_y_test(predict_y_test < 0) = -1;
predict_y_test(predict_y_test >= 0) = 1;
%Saving the predicted labels to a .mat file
save('predicted_MLP_labels_GAFeatures.mat', 'predict_y_test');
%% RBF for the best GA features:
clc;
% Defining activation functions
activation_functions = ["radbas"];
% Initializing best accuracy and the corresponding parameters
best_accuracy = 0;
best_N = zeros(1, length(activation_functions));
best_activation_function = cell(1, length(activation_functions));
% Storing accuracies for each fold initially giving the zero value
accuracies = zeros(1, 5); 
% Performing 5-fold cross-validation
for k = 1:5
    % Split data into training and validation sets
    train_indices = [1:(k-1)*110, k*110+1:550];
    valid_indices = (k-1)*110+1:k*110;
    TrainX = BestFeatureMat_GA(:, train_indices);
    ValX = BestFeatureMat_GA(:, valid_indices);
    TrainY = train_label(train_indices);
    ValY = train_label(valid_indices);
    %Creating and training RBF network
    net = newrb(TrainX, TrainY, 0, 10);          
    %Make predictions on validation set
    predict_y = sim(net, ValX);
    % Apply threshold to separate classes
    predict_y(predict_y < 0) = -1;
    predict_y(predict_y >= 0) = 1;
    % Calculate accuracy
    accuracies(k) = sum(predict_y == ValY) / length(ValY);
end        
%Calculating the average accuracy over all folds
avg_accuracy = mean(accuracies);
%Display results for each activation function
disp(['Activation function: ', activation_functions, ', Accuracy = ', num2str(avg_accuracy)]);
%Predicting RBF labels for the test data using the RBF network
predict_y_test = sim(net, BestFeatureMat_GA_test);
predict_y_test(predict_y_test <= 0) = -1;
predict_y_test(predict_y_test > 0) = 1;
%Saving the predicted labels to a .mat file
save('predicted_RBF_labels_GAFeatures.mat', 'predict_y_test');
%% Functions:
function fitness_values = evaluate_fitness(population, BestFeatureMat_GA, train_label)
    % Initialize fitness_values array
    num_individuals = size(population, 1);
    fitness_values = zeros(num_individuals, 1);

    % Loop through each individual in the population
    for i = 1:num_individuals
        % Convert binary feature set to indices of selected features
        selected_feature_indices = find(population(i, :));

        % Extract selected features from BestFeatureMat_GA matrix
        selected_features = BestFeatureMat_GA(selected_feature_indices, :);

        % Calculate fitness based on selected features
        % For example, you can use a classifier and its performance metric
        % Here, we're using accuracy as a simple fitness measure
        accuracy = calculate_accuracy(selected_features, train_label);

        % Assign fitness value to the corresponding individual
        fitness_values(i) = accuracy;
    end
end

function accuracy = calculate_accuracy(features, train_label)
    %Splitting the data into training and testing sets (80% training, 20% testing)
    cv = cvpartition(train_label, 'HoldOut', 0.2);
    idx_train = training(cv); % Indices for training set
    idx_test = test(cv); % Indices for testing set
    %Training k-NN classifier on training data
    mdl = fitcknn(features(:, idx_train)', train_label(idx_train));
    %Predict labels for testing data
    predicted_labels = predict(mdl, features(:, idx_test)');
    %Calculate accuracy
    acc = sum(predicted_labels == train_label(idx_test)) / sum(idx_test);
    accuracy = max(acc);
    %disp([accuracy]);
end

function [selected_population] = selection(population, fitness_values, num_parents)
    %Perform tournament selection
    [~, sorted_indices] = sort(fitness_values, 'descend');
    selected_population = population(sorted_indices(1:num_parents), :);
end

function offspring_population = crossover(parents, crossover_rate)
    [num_parents, num_features] = size(parents);
    num_offspring = num_parents;
    offspring_population = zeros(num_offspring, num_features);
    %Perform single-point crossover
    for i = 1:2:num_parents
        if rand() <= crossover_rate
            crossover_point = randi([1, num_features]);
            offspring1 = [parents(i, 1:crossover_point), parents(i+1, crossover_point+1:end)];
            offspring2 = [parents(i+1, 1:crossover_point), parents(i, crossover_point+1:end)];
            offspring_population(i, :) = offspring1;
            offspring_population(i+1, :) = offspring2;
        else
            offspring_population(i, :) = parents(i, :);
            offspring_population(i+1, :) = parents(i+1, :);
        end
    end
end

function mutated_population = mutation(population, mutation_rate)
    [num_individuals, num_features] = size(population);
    mutated_population = population;
    for i = 1:num_individuals
        for j = 1:num_features
            if rand() <= mutation_rate
                mutated_population(i, j) = ~mutated_population(i, j);
            end
        end
    end
end

function new_population = replace(population, offspring, fitness_values)
    %Combining the parents and offspring
    combined_population = [population; offspring];
    %Calculating the fitness values for the combined population
    combined_fitness = evaluate_fitness(combined_population);
    %Sort the combined population based on fitness values
    [~, sorted_indices] = sort(combined_fitness, 'descend');
    %Select the top individuals to form the new population
    num_parents = size(population, 1);
    new_population = combined_population(sorted_indices(1:num_parents), :);
end




































