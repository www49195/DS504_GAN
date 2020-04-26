from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, LeakyReLU
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import struct


def load_data(path):
    with open(path, 'rb') as imgpath:
        struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(-1, 784)
    return images


X_train = load_data('train-images-idx3-ubyte')
X_train = X_train.reshape(-1, 784)
rgb = 255.

X_train = X_train.astype('float32')/rgb


DATA = X_train

z_dim = 100

adam = Adam(lr=0.0002, beta_1=0.5)

g = Sequential()
g.add(Dense(256,input_dim = z_dim,activation='relu'))
g.add(LeakyReLU(0.2))
g.add(Dense(512,activation='relu'))
g.add(LeakyReLU(0.2))
g.add(Dense(1024,activation='relu'))
g.add(LeakyReLU(0.2))
g.add(Dense(784, activation="sigmoid")) 
g.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

d = Sequential()
d.add(Dense(1024, input_dim=784,activation='relu'))
g.add(LeakyReLU(0.2))
d.add(Dropout(rate=0.25))
d.add(Dense(512,activation='relu'))
g.add(LeakyReLU(0.2))
d.add(Dropout(rate=0.25))
d.add(Dense(256,activation='relu'))
g.add(LeakyReLU(0.2))
d.add(Dropout(rate=0.25))
d.add(Dense(1, activation="sigmoid")) 
d.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

d.trainable = False
inputs = Input(shape=(z_dim, ))
hidden = g(inputs)
output = d(hidden)
gan = Model(inputs, output)
gan.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
print(gan.summary())

losses = {"D":[], "G":[]}
def plot_loss(losses):
    d_loss = [v[0] for v in losses["D"]]
    g_loss = [v[0] for v in losses["G"]]

    plt.figure(figsize=(10,8))
    plt.plot(d_loss, label="Discriminator loss")
    plt.plot(g_loss, label="Generator loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("GAN_model_loss_v6.png")
    plt.close()


def plot_acc(losses):
    d_acc = [v[0] for v in losses["D"]]


    plt.figure(figsize=(10, 8))
    plt.plot(d_acc, label="Discriminator acc")
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig("GAN_model_acc_v6.png")
    plt.close()


def save_imgs(epoch):
    r, c = 8, 8
    noise = np.random.normal(0, 1, (r * c, z_dim))
    gen_imgs = g.predict(noise)

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt].reshape(28,28), cmap='gray')
            axs[i,j].axis('off')
            cnt += 1     
    fig.savefig("v6Epoch_"+"tile_"+str(epoch)+".png")
    plt.close()

def save_5by5(epoch):
    h = w = 28
    num_gen = 25
    
    z = np.random.normal(size=[num_gen, z_dim])
    generated_images = g.predict(z)
    
    # plot of generation
    n = np.sqrt(num_gen).astype(np.int32)
    I_generated = np.empty((h*n, w*n))
    for i in range(n):
        for j in range(n):
            I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = generated_images[i*n+j, :].reshape(28, 28)

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(I_generated, cmap='gray')
    plt.savefig("v6Epoch_"+"5by5_"+str(epoch)+".png")
    plt.close    


def train(epochs=1, BATCH_SIZE=128,save_interval=10):
    batchCount = int(DATA.shape[0] / BATCH_SIZE)
    print('Epochs:', epochs)
    print('Batch size:', BATCH_SIZE)
    print('Batches per epoch:', batchCount)
    print("------------------------------")
    for e in (range(1, epochs+1)):       
        print("Epoch:",e)
        for _ in range(batchCount):
            idxs = np.random.randint(0, DATA.shape[0], size=BATCH_SIZE)
            image_batch = DATA[idxs]

            noise = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))

            generated_images = g.predict(noise)
            X = np.concatenate((image_batch, generated_images))

            y = np.zeros(2*BATCH_SIZE)
            y[:BATCH_SIZE] = 1 

            d.trainable = True
            d_loss = d.train_on_batch(X, y)
            noise = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))
            y2 = np.ones(BATCH_SIZE)
            d.trainable = False
            g_loss = gan.train_on_batch(noise, y2)


        losses["G"].append(g_loss)
        losses["D"].append(d_loss)

        if e % save_interval == 0:
            save_imgs(e)
            save_5by5(e)
        
train(epochs=200, BATCH_SIZE=128,save_interval=10)
plot_loss(losses)

model_json = g.to_json()
with open("generator.json", "w") as json_file:
    json_file.write(model_json)
g.save_weights("generator.h5")
print("Saved model to disk")
