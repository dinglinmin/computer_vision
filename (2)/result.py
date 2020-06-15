eval_loss = 0
	eval_acc = 0
	model.eval() # 将模型改为预测模式
	for step, (image, label) in enumerate(test_data):
	    image = Variable(image)
	    label = Variable(label)
	        
	    out = model(image)
	    
	    loss = criterion(out, label)
	    # 记录误差
	    eval_loss += loss.data
	    # 记录准确率
	    _, pred = out.max(1)
	    num_correct = (pred == label).sum().item()
	    acc = num_correct / image.shape[0]
	    eval_acc += acc
	
	print('Test Loss: {:.6f}, Test Accuracy: {:.6f}'.format(eval_loss/len(test_data), eval_acc/len(test_data)))
