import torch
from tqdm import tqdm
from liveplotloss import PlotLosses

from DatasetFetcher import DatasetFetcher
from project1_model import ResNet, BasicBlock 

if __name__ == "__main__":
    # Check available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Fetching Dataset
    df = DatasetFetcher(dataset="CIFAR10", batch_size=64)
    df.addHorizontalFlipping()
    df.addRandomCrop(size=32, padding=3)
    df.addNormalizer()
    trainLoader, testLoader = df.getLoaders()
    
    # Get Model
    model = ResNet(BasicBlock, 4, [3, 3, 3, 3])
    model = model.to(device)
    
    # Defining Loss Function, Learning Rate, Weight Decay, Optimizer) 
    lossFunction = torch.nn.CrossEntropyLoss()
    learningRate = 0.001
    weightDecay = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, weight_decay=weightDecay)
    print(model.eval())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Trainable Parameters : %s"%(trainable_parameters))
    if trainable_parameters > 5*(10**6):
        raise Exception("Model not under budget!")
    
    # Setting up training
    EPOCHS=1
    trainingLoss = []
    testingLoss = []
    trainingAccuracy = []
    testingAccuracy = []
    
    liveLoss = PlotLosses()
    # Training
    for i in tqdm(range(EPOCHS)):
        liveLossLogs = {}
        for phase in ['train', 'test']:
            if phase == "train":
                loader = trainLoader
                model.train()
                optimizer.zero_grad()
            else:
                loader = testLoader
                model.eval()
            runningLoss = 0.0
            runningCorrects = 0
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                loss = lossFunction(output, labels)
                predicted_labels = torch.argmax(output, dim=1)
                runningLoss = loss.item()*images.size(0)
                runningCorrects += torch.sum(predicted_labels == labels).float().item()
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                break
            epochLoss = runningLoss/len(loader.dataset)
            epochAccuracy = runningCorrects/len(loader.dataset)
            if phase == "train":
                trainingLoss.append(epochLoss)
                trainingAccuracy.append(epochAccuracy)
            else:
                testingLoss.append(epochLoss)
                testingAccuracy.append(epochAccuracy)
        liveLossLogs["loss"] = trainingLoss[-1]
        liveLossLogs["val_loss"] = testingLoss[-1]
        liveLossLogs["accuracy"] = trainingAccuracy[-1]
        liveLossLogs["val_accuracy"] = testingAccuracy[-1]
        liveLoss.update(liveLossLogs)
        liveLoss.update()
            
    model.saveToDisk()
    
