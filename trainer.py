import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import json
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
from polygon_net import PolygonDataset, PolyGonNet
from torchmetrics.functional import structural_similarity_index_measure as ssim
import wandb
from torch.amp import GradScaler, autocast

wandb.init(
	project='UNet Polygon color filling',
	name='run_2',
	config={
		"epochs": 100000,
		"batch_size": 16,
		"lr": 1e-3,
		"optimizer": 'AdamW',
		"model params": 1925968
	}
)

dataset = PolygonDataset(mode='train', augmentation=True)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, pin_memory=True, num_workers=4)

def train(
	epochs=100000,
	lr=1e-3,
	weight_decay=1e-5,
	patience=300,
	threshold=1e-5,
	plot_freq=500,
	save_freq=500,
	img_log_freq=500
	):
	epochs = epochs
	polygon_net = PolyGonNet()
	optimizer = optim.AdamW(
		polygon_net.parameters(),
		lr=lr,
		weight_decay=weight_decay
	)

	training_losses = []
	val_losses = []

	patience = patience
	threshold = threshold
	epoch_wo_improvement = 0
	plot_freq = plot_freq
	save_freq = save_freq
	img_log_freq = img_log_freq
	avg_val_loss = np.inf

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	polygon_net = polygon_net.to(device)

	scaler = GradScaler('cuda')

	for epoch in range(epochs):
		prev_avg_training_loss = avg_training_loss if epoch > 0 else 0
		avg_training_loss = 0
		for j, batch in enumerate(train_loader):
			polygon_net.train()
			(image, coloridx), target = batch
			image = image.to(device)
			coloridx = coloridx.to(device)
			target = target.to(device)
			
			with autocast('cuda'):
				pred = polygon_net(image, coloridx)
			
				recon_loss = F.mse_loss(pred, target)
				ssim_loss = 1 - ssim(pred, target)
				total_loss = recon_loss + 0.1 * ssim_loss
			
				optimizer.zero_grad()
				scaler.scale(total_loss).backward()
				scaler.step(optimizer)
				scaler.update()
		
			avg_training_loss += (1 / (j + 1)) * (total_loss.item() - avg_training_loss)

		training_losses.append(avg_training_loss)
		print(f'\nstep {epoch+1} training loss => {avg_training_loss}')
		
		avg_val_loss = 0
		for j, val_batch in enumerate(val_loader):
			polygon_net.eval()
			(image, coloridx), target = val_batch
			image = image.to(device)
			coloridx = coloridx.to(device)
			target = target.to(device)

			with torch.no_grad():
				pred = polygon_net(image, coloridx)
				recon_loss = F.mse_loss(pred, target)
				ssim_loss = 1 - ssim(pred, target)
				total_loss = recon_loss + 0.1 * ssim_loss

				if (epoch + 1) % img_log_freq == 0:
					wandb.log({
						"epoch": epoch+1,
						"input img": wandb.Image(image.cpu().squeeze(0).permute(1, 2, 0), caption="Input"),
						"target_img": wandb.Image(target.cpu().squeeze(0).permute(1, 2, 0), caption="Target"),
						"predicted_img": wandb.Image(pred.cpu().squeeze(0).permute(1, 2, 0), caption="Prediction")
					})

			avg_val_loss += (1 / (j + 1)) * (total_loss.item() - avg_val_loss)

		val_losses.append(avg_val_loss)
		print(f'step {epoch+1} validation loss => {avg_val_loss}')

		wandb.log({
			"epoch": epoch+1, 
			"avg_train_loss": avg_training_loss, 
			"avg_val_loss": avg_val_loss
		})

		if (epoch + 1) % save_freq == 0:
			torch.save(polygon_net.state_dict(), os.path.join(os.getcwd(), 'Models', f'polygon_model_{epoch+1}.pth'))

		if abs(prev_avg_training_loss - avg_training_loss) < threshold:
			epoch_wo_improvement += 1
		elif 0 < epoch_wo_improvement <= patience:
			epoch_wo_improvement = 0
		elif epoch_wo_improvement > patience:
			print("No improvement seen, terminating to avoid overfitting!")
			torch.save(polygon_net.state_dict(), r'Models\polygon_model_noimprov.pth')
			break

		if (epoch + 1) % plot_freq == 0:
			plt.figure(figsize=(10, 5))
			plt.plot(range((epoch+1)-plot_freq, (epoch+1)), training_losses[(epoch+1)-plot_freq:(epoch+1)], label='Training Loss')
			plt.plot(range((epoch+1)-plot_freq, (epoch+1)), val_losses[(epoch+1)-plot_freq:(epoch+1)], label='Validation Loss')
			plt.xlabel('Epochs')
			plt.ylabel('Reconstruction Loss')
			plt.legend()
			plt.grid(True)
			plt.tight_layout()
			plt.show()

	torch.save(polygon_net.state_dict(), r'Models\final_polygon_model.pth' )

if __name__ == '__main__':
	train()	

