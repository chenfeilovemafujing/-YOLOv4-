import torch
import torch.nn as nn
import math



class dehaze_net(nn.Module):

	def __init__(self):
		super(dehaze_net, self).__init__()

		self.relu = nn.ReLU(inplace=True)
	
		self.e_conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, padding=0, bias=True)
		self.e_conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=True)
		self.e_conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, padding=2, bias=True)
		self.e_conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, padding=3, bias=True)
		self.e_conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, padding=1, bias=True)
		self.pooling1 = nn.MaxPool2d(kernel_size=4)
		self.pooling2 = nn.MaxPool2d(kernel_size=8)
		self.pooling3 = nn.MaxPool2d(kernel_size=16)
		self.pooling4 = nn.MaxPool2d(kernel_size=32)
		self.e_conv6 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
		self.upsample1 = nn.functional.interpolate
		self.upsample2 = nn.functional.interpolate
		self.upsample3 = nn.functional.interpolate
		self.upsample4 = nn.functional.interpolate
		self.e_conv7 = nn.Conv2d(in_channels=7, out_channels=3, kernel_size=3, padding=1)
		
	def forward(self, x):
		source = []
		source.append(x)

		x1 = self.relu(self.e_conv1(x))
		x2 = self.relu(self.e_conv2(x1))

		concat1 = torch.cat((x1, x2), 1)
		x3 = self.relu(self.e_conv3(concat1))

		concat2 = torch.cat((x2, x3), 1)
		x4 = self.relu(self.e_conv4(concat2))

		concat3 = torch.cat((x1, x2, x3, x4),1)
		x5 = self.relu(self.e_conv5(concat3))

		p1 = self.pooling1(x5)
		p1 = self.relu(self.e_conv6(p1))
		p2 = self.pooling2(x5)
		p2 = self.relu(self.e_conv6(p2))
		p3 = self.pooling3(x5)
		p3 = self.relu(self.e_conv6(p3))
		p4 = self.pooling4(x5)
		p4 = self.relu(self.e_conv6(p4))

		u1 = self.upsample1(p1, scale_factor=4, mode='bilinear', align_corners=True)
		u2 = self.upsample2(p2, scale_factor=8, mode='bilinear', align_corners=True)
		u3 = self.upsample3(p3, scale_factor=16, mode='bilinear', align_corners=True)
		u4 = self.upsample4(p4, scale_factor=32, mode='bilinear', align_corners=True)

		concat4 = torch.cat((x5, u1, u2, u3, u4), 1)

		x6 = self.relu(self.e_conv7(concat4))

		clean_image = self.relu((x6 * x) - x6 + 1)
		
		return clean_image




			

			
			






