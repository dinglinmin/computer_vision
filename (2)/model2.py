class CNN(nn.Module):
	    def __init__(self):
	        super(CNN, self).__init__()
	        self.layer1 = nn.Sequential(    		# [1, 28, 28]
	            nn.Conv2d(1, 16, kernel_size=3),  	# [16, 26, 26]
	            nn.BatchNorm2d(16),
	            nn.ReLU()
	        )
	        
	        self.layer2 = nn.Sequential( 			 # [16, 26, 26]
	            nn.Conv2d(16, 32, kernel_size=3),  	 # [32, 24, 24]
	            nn.BatchNorm2d(32),
	            nn.ReLU(),
	            nn.MaxPool2d(kernel_size=2, stride=2)	# [32, 12, 12]
	        )
	        
	        self.layer3 = nn.Sequential(     		# [32, 12, 12]
	            nn.Conv2d(32, 64, kernel_size=3),   # [64, 10, 10]
	            nn.BatchNorm2d(64),
	            nn.ReLU()
	        )
	        
	        self.layer4 = nn.Sequential(  			# [64, 10, 10]
	            nn.Conv2d(64, 128, kernel_size=3),  # [128, 8, 8]
	            nn.BatchNorm2d(128),
	            nn.ReLU(),
	            nn.MaxPool2d(kernel_size=2, stride=2) 	# [128, 4, 4]
	        )
	        
	        self.fc = nn.Sequential(
	            nn.Linear(128*4*4, 1024),
	            nn.ReLU(),
	            nn.Linear(1024, 128),
	            nn.ReLU(),
	            nn.Linear(128, 10),
	        )
	        
	    def forward(self, x):
	        x = self.layer1(x)
	        x = self.layer2(x)
	        x = self.layer3(x)
	        x = self.layer4(x)
	        x = x.view(x.size(0), -1) 
	        output = self.fc(x)
	        
	        return output

model = CNN()
	model.parameters