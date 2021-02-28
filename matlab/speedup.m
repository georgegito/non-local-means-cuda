close all;
%% read data
v1_times = xlsread('../report/benchmark/v1/cpu.xlsx','B2:D4')
v2_times = xlsread('../report/benchmark/v2/gpu_gm.xlsx','B2:D4')
v3_times = xlsread('../report/benchmark/v3/gpu_sm.xlsx','B2:D4')

%% house image (64*64) speedup
speedup_3_64 = [1, v1_times(1, 1) / v2_times(1, 1),  v1_times(1, 1) / v3_times(1, 1)];
speedup_5_64 = [1, v1_times(1, 2) / v2_times(1, 2),  v1_times(1, 2) / v3_times(1, 2)];
speedup_7_64 = [1, v1_times(1, 3) / v2_times(1, 3),  v1_times(1, 3) / v3_times(1, 3)];

%% flower image (128*128) speedup
speedup_3_128 = [1, v1_times(2, 1) / v2_times(2, 1),  v1_times(2, 1) / v3_times(2, 1)];
speedup_5_128 = [1, v1_times(2, 2) / v2_times(2, 2),  v1_times(2, 2) / v3_times(2, 2)];
speedup_7_128 = [1, v1_times(2, 3) / v2_times(2, 3),  v1_times(2, 3) / v3_times(2, 3)];

%% lena image (256*256) speedup
speedup_3_256 = [1, v1_times(3, 1) / v2_times(3, 1),  v1_times(3, 1) / v3_times(3, 1)];
speedup_5_256 = [1, v1_times(3, 2) / v2_times(3, 2),  v1_times(3, 2) / v3_times(3, 2)];
speedup_7_256 = [1, v1_times(3, 3) / v2_times(3, 3),  v1_times(3, 3) / v3_times(3, 3)];

%% plots
figure;
grid on;
hold on;
plot(speedup_3_64, '-o', 'MarkerFaceColor', 'black', "Linewidth", 2);
plot(speedup_3_128, '-o', 'MarkerFaceColor', 'black', "Linewidth", 2);
plot(speedup_3_256, '-o', 'MarkerFaceColor', 'black', "Linewidth", 2);
title("patchSize = 3");
ylabel("speedup t(v_1) / t(v_i)");
legend("n = 64", "n = 128", "n = 256");
xticklabels({'version 1', ' ','version 2', ' ', 'version 3'});

figure;
grid on;
hold on;
plot(speedup_5_64, '-o', 'MarkerFaceColor', 'black', "Linewidth", 2);
plot(speedup_5_128, '-o', 'MarkerFaceColor', 'black', "Linewidth", 2);
plot(speedup_5_256, '-o', 'MarkerFaceColor', 'black', "Linewidth", 2);
title("patchSize = 5");
ylabel("speedup t(v_1) / t(v_i)");
legend("n = 64", "n = 128", "n = 256");
xticklabels({'version 1', ' ','version 2', ' ', 'version 3'});

figure;
grid on;
hold on;
plot(speedup_7_64, '-o', 'MarkerFaceColor', 'black', "Linewidth", 2);
plot(speedup_7_128, '-o', 'MarkerFaceColor', 'black', "Linewidth", 2);
plot(speedup_7_256, '-o', 'MarkerFaceColor', 'black', "Linewidth", 2);
title("patchSize = 7");
ylabel("speedup t(v_1) / t(v_i)");
legend("n = 64", "n = 128", "n = 256");
xticklabels({'version 1', ' ','version 2', ' ', 'version 3'});

figure;
grid on;
hold on;
plot(speedup_3_64, '-o', 'MarkerFaceColor', 'black', "Linewidth", 2);
plot(speedup_5_64, '-o', 'MarkerFaceColor', 'black', "Linewidth", 2);
plot(speedup_7_64, '-o', 'MarkerFaceColor', 'black', "Linewidth", 2);
title("n = 64");
ylabel("speedup t(v_1) / t(v_i)");
legend("patchSize = 3", "patchSize = 5", "patchSize = 7");
xticklabels({'version 1', ' ','version 2', ' ', 'version 3'});

figure;
grid on;
hold on;
plot(speedup_3_128, '-o', 'MarkerFaceColor', 'black', "Linewidth", 2);
plot(speedup_5_128, '-o', 'MarkerFaceColor', 'black', "Linewidth", 2);
plot(speedup_7_128, '-o', 'MarkerFaceColor', 'black', "Linewidth", 2);
title("n = 128");
ylabel("speedup t(v_1) / t(v_i)");
legend("patchSize = 3", "patchSize = 5", "patchSize = 7");
xticklabels({'version 1', ' ','version 2', ' ', 'version 3'});

figure;
grid on;
hold on;
plot(speedup_3_256, '-o', 'MarkerFaceColor', 'black', "Linewidth", 2);
plot(speedup_5_256, '-o', 'MarkerFaceColor', 'black', "Linewidth", 2);
plot(speedup_7_256, '-o', 'MarkerFaceColor', 'black', "Linewidth", 2);
title("n = 256");
ylabel("speedup t(v_1) / t(v_i)");
legend("patchSize = 3", "patchSize = 5", "patchSize = 7");
xticklabels({'version 1', ' ','version 2', ' ', 'version 3'});


