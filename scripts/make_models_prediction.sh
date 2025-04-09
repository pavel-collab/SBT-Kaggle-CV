#!/bin/bash

for dir in ./saved_models/*; do
    for model_path in $dir/*; do
        # Извлекаем родительскую директорию
        parent_directory="${model_path%/*}"
        # Извлекаем имя родительской директории
        parent_name="${parent_directory##*/}"
        
        filename=$(basename "$model_path")
        model_name=$(echo "$filename" | sed 's/best_model_//; s/.pth//')

        echo "Predict on model $model_name with head $parent_name"
        python3 ./src/predict.py -m $model_path
    done
done