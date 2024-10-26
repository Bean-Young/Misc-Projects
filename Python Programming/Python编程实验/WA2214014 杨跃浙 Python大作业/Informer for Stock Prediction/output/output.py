import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

setting = 'informer_custom_ftMS_sl400_ll200_pl100_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_1'
pred = np.load('E:/Python/Project/Informer for Stock Prediction/results/' + setting + '/pred.npy')
true = np.load('E:/Python/Project/Informer for Stock Prediction/results/' + setting + '/true.npy')
print(pred.shape)
print(true.shape)

plt.figure()
plt.plot(true[0, :, -1], label='GroundTruth')
plt.plot(pred[0, :, -1], label='Prediction')
plt.legend()
plt.savefig('Stock Prediction.png')

def watermark_Image(img_path,output_path, text, pos):
    img = Image.open(img_path)
    drawing = ImageDraw.Draw(img)
    black = (0, 0, 0)
    textsize=25
    ft=ImageFont.truetype(font='E:/Python/Project/Informer for Stock Prediction/Gidole-master/GidoleFont/Gidole-Regular.ttf',size=textsize)
    drawing.text(pos, text, fill=black,font=ft,align="center")
    jpg=Image.open('E:/Python/Project/Informer for Stock Prediction/AI.jpg')
    jpg_x=img.width-jpg.width-20
    jpg_y=0
    img.paste(jpg,(jpg_x,jpg_y))
    img.show()
    img.save(output_path)


img = 'Stock Prediction.png'
watermark_Image(img, 'E:/Python/Project/Informer for Stock Prediction/Stock Prediction.png',u'AHUniversity\nWA2214014 YangYuezhe', pos=(20, 0))

#plt.show()
