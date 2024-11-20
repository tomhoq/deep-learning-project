from numpy import ceil
import torch
import torch.nn.functional as F

@torch.no_grad()
def validation(model: torch.nn.Module, loss_function, valid_loader, device):
    valid_loss = 0
    num_correct = 0
    tot = 0 

    model.eval()
    for inputs, targets in valid_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        loss = loss_function(outputs, targets)
        valid_loss += loss * valid_loader.batch_size

        # correct = torch.argmax(F.softmax(outputs, dim=1), dim=1) == targets
        correct = (outputs >= 0.5) == targets
        tot += correct.shape[0]
        num_correct += correct.sum()

    # Normalize
    valid_loss /= len(valid_loader.dataset)
    valid_accuracy = num_correct / tot
    
    print('    Valid loss: {:.5f}, Valid accuracy: {:.5f}'.format(valid_loss, valid_accuracy))

    return {
        'valid_loss': valid_loss.item(), 
        'valid_accuracy': valid_accuracy.item(),
    }
