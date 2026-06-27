set terminal pngcairo size 800,800 enhanced font 'DejaVu Sans,12'
set output 'training_plot.png'

set multiplot layout 2,1

# График Loss
set title 'Функция потерь (Loss)' font 'DejaVu Sans,14'
set xlabel 'Эпоха' font 'DejaVu Sans,12'
set ylabel 'Loss' font 'DejaVu Sans,12'
set grid
set key top right
plot 'training.log' using 1:2 with lines lw 2.5 lc rgb '#4CAF50' title 'Loss'

# График Accuracy
set title 'Точность (Accuracy)' font 'DejaVu Sans,14'
set xlabel 'Эпоха' font 'DejaVu Sans,12'
set ylabel 'Accuracy, %' font 'DejaVu Sans,12'
set grid
set key bottom right
plot 'training.log' using 1:3 with lines lw 2.5 lc rgb '#2196F3' title 'Accuracy'

unset multiplot
