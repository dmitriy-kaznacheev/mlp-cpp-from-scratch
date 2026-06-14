#!/usr/bin/env python3

import struct
import subprocess
import os

os.chdir('../build')

with open('../data/t10k-images-idx3-ubyte', 'rb') as img_file, \
     open('../data/t10k-labels-idx1-ubyte', 'rb') as lbl_file:
    
    img_file.read(16)
    lbl_file.read(8)
    
    correct = 0
    total = 1000
    
    for i in range(total):
        label = struct.unpack('B', lbl_file.read(1))[0]
        img_data = img_file.read(784)
        
        filename = f'test_{i}.pgm'
        with open(filename, 'wb') as out:
            out.write(b'P5\n28 28\n255\n')
            out.write(bytes([255 - b for b in img_data]))  # с инверсией цветов
        
        result = subprocess.run(
            ['./app/console_app/mnist_console', 'predict', 'model.bin', filename],
            capture_output=True, text=True
        )
        
        for line in result.stdout.split('\n'):
            if 'Predicted digit:' in line:
                predicted = int(line.split(':')[1].strip())
                if predicted == label:
                    correct += 1
                else:
                    print(f'[Err] {filename}: {label} -> {predicted}')
                break
        
        os.remove(filename)
    
    print(f'\nAccuracy on {total} test images: {correct}/{total} = {correct/total*100:.1f}%')
