import torch


def validation(model: torch.nn.Module, loss_function, valid_loader, device):
    valid_loss = 0
    valid_accuracy = 0

    model.eval()
    
    for inputs, targets in valid_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        valid_loss += loss * valid_loader.batch_size
        valid_accuracy += (torch.argmax(outputs, dim=1) == targets).sum()
    
    print('    Valid loss: {:.5f}, Valid accuracy: {:.5f}'.format(valid_loss, valid_accuracy))

    return {
        'valid_loss': valid_loss, 
        'valid_accuracy': valid_accuracy,
    }