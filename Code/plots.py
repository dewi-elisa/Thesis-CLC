import numpy as np
import matplotlib.pyplot as plt

x_words = np.arange(5, 50+1, 5)
loss_words_10 = [0.42260774970054626, 0.29506322741508484, 0.3109319806098938, 0.50328528881073, 0.35804682970046997, 0.5930567979812622, 0.36207857728004456, 0.3256344497203827, 0.32602939009666443, 0.2771590054035187]
loss_words_5 = [0.30331122875213623, 0.34467238187789917, 0.462980180978775, 0.35684871673583984, 0.327193945646286, 0.383816123008728, 0.3710336685180664, 0.32320964336395264, 0.34782448410987854, 0.3316417336463928]

plt.figure()
plt.plot(x_words, loss_words_10, label='10')
plt.plot(x_words, loss_words_5, label='5')
plt.legend(title='max_length')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('images/loss_words.png')

x_images = np.arange(5, 25+1, 5)
loss_images_10 = [0.8688374161720276, 0.4721875786781311, 0.5944969058036804, 0.46159815788269043, 0.46575814485549927]
loss_images_5 = [0.688713550567627, 0.4851745367050171, 0.4736078679561615, 0.5496516227722168, 0.492509126663208]

plt.figure()
plt.plot(x_images, loss_images_10, label='10')
plt.plot(x_images, loss_images_5, label='5')
plt.xticks(range(5,25+1, 5))
plt.legend(title='max_length')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('images/loss_images.png')
