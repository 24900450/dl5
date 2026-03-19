# dl5
```python
## Step 2: Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  #   Take the output of the last time step
        return out

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


## Step 3: Train the Model
from functools import total_ordering
def train_model(model,train_loader,criterian,optimizer,epochs=20):
  model.train()
  train_losses = []
  for epoch in range(epochs):
      total_loss = 0
      epoch_loss = 0
      for x_batch, y_batch in train_loader:
          x_batch, y_batch = x_batch.to(device), y_batch.to(device)
          optimizer.zero_grad()
          outputs = model(x_batch)
          loss = criterion(outputs, y_batch)
          loss.backward()
          optimizer.step()
          total_loss += loss.item()
      train_losses.append(total_loss/len(train_loader))
      print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')
  # Plot training loss
  print('Name:SURYANARAYANAN T')
  print('Register Number:212224040341')
  plt.plot(train_losses, label='Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('MSE Loss')
  plt.title('Training Loss Over Epochs')
  plt.legend()
  plt.show()

train_model(model,train_loader,criterion,optimizer,epochs=20)



```
