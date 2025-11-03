from classification import main
import sys
import matplotlib.pyplot as plt
import numpy as np

small_subset = True
num_epochs = 8
lr = 1e-4
batch_size = 64
device = 'cuda'
model = 'RoBERTa-base'
type = 'full'

options = [ '--small_subset', 
			'--device', device,
			'--model', model,
			'--batch_size', str(batch_size),
			'--lr', str(lr),
			'--num_epochs', str(num_epochs),
			'--type', type
		]
print(">>>>>>>>>>>>>>>>>>>     Q1")
sys.argv = options
val_accuracy, test_accuracy = main()
print("test_accuracy: ", test_accuracy)


type = 'head'
options = [ '--small_subset', 
			'--device', device,
			'--model', model,
			'--batch_size', str(batch_size),
			'--lr', str(lr),
			'--num_epochs', str(num_epochs),
			'--type', type
		]
print(">>>>>>>>>>>>>>>>>>>     Q2")
sys.argv = options
val_accuracy, test_accuracy = main()
print("test_accuracy: ", test_accuracy)


type = 'prefix'
options = [ '--small_subset', 
			'--device', device,
			'--model', model,
			'--batch_size', str(batch_size),
			'--lr', str(lr),
			'--num_epochs', str(num_epochs),
			'--type', type
		]
print(">>>>>>>>>>>>>>>>>>>     Q3")
sys.argv = options
val_accuracy, test_accuracy = main()
print("test_accuracy: ", test_accuracy)