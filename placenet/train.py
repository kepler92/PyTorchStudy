import densenet

model = densenet.Models()
for i in range(300):
    model.train()
model.test()