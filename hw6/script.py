from base_classification import main
import sys
import matplotlib.pyplot as plt
import numpy as np

small_subset = True
# num_epochs = 7
# lr = 1e-4
batch_size = 64
device = 'cuda'
# model = 'distilbert-base-uncased'

num_epochs_arr = [7, 9]
lr_arr = [1e-4, 5e-4, 1e-3]
models_arr = ['BERT-base-uncased', 'RoBERTa-base']
results = []

for model in models_arr:
	for num_epochs in num_epochs_arr:
		for lr in lr_arr:
			options = [ '--small_subset', 
						'--device', device,
						'--model', model,
						'--batch_size', str(batch_size),
						'--lr', str(lr),
						'--num_epochs', str(num_epochs)]
			sys.argv = options
			val_accuracy, test_accuracy = main()

			results.append((model, num_epochs, lr, val_accuracy, test_accuracy))

# --- Plot results ---
labels = [f" {m} model, {e} ep, lr={lr}" for m, e, lr, _, _ in results]
val_accuracies = [v for _, _, _, v, _ in results]
test_accuracies = [t for _, _, _, _, t in results]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, val_accuracies, width, label='Validation Accuracy')
rects2 = ax.bar(x + width/2, test_accuracies, width, label='Test Accuracy')

ax.set_xlabel('Configuration')
ax.set_ylabel('Accuracy')
ax.set_title('Validation vs Test Accuracy for Each Hyperparameter Setting')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig("accuracy_barplot.png", dpi=300)
plt.show()