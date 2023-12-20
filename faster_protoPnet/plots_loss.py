import re
import matplotlib.pyplot as plt


file_path = '/home/up202003072/Documents/faster_protoPnet/sbatch-49346-python3-train.py.txt' 
with open(file_path, 'r') as file:
    file_contents = file.read()

epoch_pattern = r'Epoch:\n(\d+)'
loss_pattern = r"'total_loss': ([\d.]+)"
epochs = re.findall(epoch_pattern, file_contents)
losses = re.findall(loss_pattern, file_contents)
epochs = [int(epoch) for epoch in epochs]
losses = [float(loss) for loss in losses]

plt.plot(epochs, losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Total Loss per Epoch')
plt.grid(True)
plt.show()
