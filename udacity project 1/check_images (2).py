#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/check_images.py                                      
# PROGRAMMER: Rahul Ghosh
# DATE CREATED: 02/05/2020                                  
# REVISED DATE: 
# PURPOSE: Classifies pet images using a pretrained CNN model, compares these
#          classifications to the true identity of the pets in the images, and
#          summarizes how well the CNN performed on the image classification task. 
#          Note that the true identity of the pet (or object) in the image is 
#          indicated by the filename of the image. Therefore, your program must
#          first extract the pet image label from the filename before
#          classifying the images using the pretrained CNN model. With this 
#          program we will be comparing the performance of 3 different CNN model
#          architectures to determine which provides the 'best' classification.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
from time import time, sleep
from print_functions_for_lab_checks import *
from prettytable import PrettyTable
# Imports functions created for this program
from get_input_args import get_input_args
from get_pet_labels import get_pet_labels
from classify_images import classify_images
from adjust_results4_isadog import adjust_results4_isadog
from calculates_results_stats import calculates_results_stats
from print_results import print_results


# Main program function defined below
def main():
    start_time = time() # start_time from time module (0)
    #1
    in_arg = get_input_args() #read command line args, using argparse mod(1)
    check_command_line_arguments(in_arg) #prints command line arguments (1)
    #2
    results = get_pet_labels(in_arg.dir) #creates dictionary with key: file_name & value: [pet_label]
    check_creating_pet_image_labels(results) # prints 10 of key value pairs
    #3
    classify_images(in_arg.dir, results, in_arg.arch)
    check_classifying_images(results)
    adjust_results4_isadog(results, in_arg.dogfile)
    check_classifying_labels_as_dogs(results)
    results_stats = calculates_results_stats(results)
    check_calculating_results(results, results_stats)
    print_results(results, results_stats, in_arg.arch, True, True)
  
    #python check_images.py --dir pet_images/ --arch resnet --dogfile dognames.txt --> 6 seconds, but pct_correct_dogs is wrong
    #python check_images.py --dir pet_images/ --arch alexnet --dogfile dognames.txt --> 3 seconds
    #python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt --> 37 seconds
    
    end_time = time()
    tot_time = end_time - start_time 
    print("\n** Total Elapsed Runtime:", str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)))
    
    table1 = PrettyTable()
    table1.field_names = ["# Total Images", "# Dog Images", "# Not-a-Dog Images"]
    table1.add_row([40, 30, 10])
    print(table1)
    print("\n\n\n")
    table2 = PrettyTable()
    table2.field_names = ["CNN Model Architecture: ", "% Not-a-Dog Correct", "% Dogs Corrects", "% Breeds Correct", "% Match Labels", "Runtime (seconds)"]
    table2.add_row(["ResNet", "90%", "100%", "90%", "82.5%", 6])
    table2.add_row(["AlexNet", "100%", "100%", "80%", "75%", 3])
    table2.add_row(["VGG", "100%", "100%", "93.3%", "87.5%", 35])
    print(table2)
    
    print("The model VGG was the one that was able to classify 'dogs' and 'not-a-dog' with 100% accuracy and had the best performance regarding breed classification with 93.3% accuracy. The Model AlexNet was the most efficient with the fastest runtime at only 3 seconds but still images 100% accuracy for identifying dogs correctly")
    

   
# Call to main function to run the program
if __name__ == "__main__":
    main()