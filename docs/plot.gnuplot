set terminal pngcairo size 800,400 enhanced font 'DejaVu Sans,12'
set output 'training_plot.png'

set multiplot layout 1,2

set title 'Функция потерь (Loss)'
set xlabel 'Эпоха'
set ylabel 'Loss'
set grid
plot 'training.log' using 1:2 with lines lw 2 lc rgb '#4CAF50' title 'Loss'

set title 'Точность (Accuracy)'
set xlabel 'Эпоха'
set ylabel 'Accuracy, %'
set grid
plot 'training.log' using 1:3 with lines lw 2 lc rgb '#2196F3' title 'Accuracy'

unset multiplot
