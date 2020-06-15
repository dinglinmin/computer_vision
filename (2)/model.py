class simpleCNN(nn.Module):
	    def __init__(self):
	        super(simpleCNN, self).__init__()
	        self.layer1 = nn.Sequential(  # 1, 28, 28
	            nn.Conv2d(1, 16, 5, 1, 2),   # 卷积层,输入深度为1,输出深度16,卷积核5*5,步长1,padding=(kernel_size-1)/2如果stride=1
	            nn.ReLU(),  # 激活层
	            nn.MaxPool2d(kernel_size=2)  # 池化层
	        )  # 输出: 16, 14, 14
	        
	        self.layer2 = nn.Sequential(   # 全连接层 
	            nn.Linear(16*14*14, 10)   # 32*7*7
	        )
	        
	    def forward(self, x):
	        x = self.layer1(x)
	        x = x.view(x.size(0), -1)   # 多维展开
	        output = self.layer2(x)
	        
	        return output
			
model = simpleCNN()
	
model.parameters