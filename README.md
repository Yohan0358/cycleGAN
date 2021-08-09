```python
$python3 train.py --total_epochs 150\
                  --lr 2e-4\
                  --batch_size 1\
                  --save_epoch 10\
                  --continue_train 'on'\  #['on', 'off']
                  --save_dir './save_model\
                  --weight_cycle 10\
                  --wegiht_identity 5\
                  --img_path './dataset'
```
## Result
### Monet to Photo
![](https://github.com/Yohan0358/cycleGAN/blob/master/output/A2B_130.jpg?raw=true)
![](https://github.com/Yohan0358/cycleGAN/blob/master/output/A2B_150.jpg?raw=true)

### Photo to Monet
![](https://github.com/Yohan0358/cycleGAN/blob/master/output/B2A_130.jpg?raw=true)
![](https://github.com/Yohan0358/cycleGAN/blob/master/output/B2A_150.jpg?raw=true)

---
# cycleGAN
- pix2pix는 translate하려는 데이터가 쌍으로 있어야 학습이 가능
  - 데이터 얻기가 어렵고 비용이 많이 드는 단점
- unpair dataset에서 데이터의 특징(스타일)만 가져와 translation을 할 수 없을까? → cycleGAN의 핵심 아이디어

![](https://raw.githubusercontent.com/dimitreOliveira/MachineLearning/master/Kaggle/I%E2%80%99m%20Something%20of%20a%20Painter%20Myself/cyclegan_horse-zebra.jpg)

## 2. cycleGAN의 구조
![](https://hardikbansal.github.io/CycleGANBlog/images/model.jpg)
- 2개의 Generator와 2개의 Discriminator로 구성
  - Generator : X-domain to Y-domain(Gx), Y-domain to X-domain(Gy)
  - Discriminator : Dx, Dy

### 2.1 Network
![](https://www.lyrn.ai/wp-content/uploads/2019/01/CycleGAN-arch.png)

- Generator
  - 단순 Encoder-Decoder모델은 downsampling 과정에서 중간 정보가 손실됨
  - 이를 보완하기 위해 중간 정보를 upsampling layer에 전달하는 UNet이 Encoder-Decoder모델보다는 유리하지만, Bottleneck 구조는 정보 손실이 발생할 수 밖에 없음
  - *ResNet 관련 내용 추가*
  
- Discriminator
  - pix2pix와 동일하게 patch GAN(30x30)을 사용한 구조

### 2.1 Loss function
![](https://miro.medium.com/max/2000/1*YOhXT4gecyVgpMQHsrIvsw.png)
- X → Y → X로 2번의 Generator가 생성하는 이미지는 원본 이미지와 같아야 함 → cycle consistency loss
- Dx(x) = 1, Dx(Gx(x)) = 0, Dy(y) = 1, Dy(Gy(y)) = 0 ↔︎ Dx(Gy(y)) = 1, Dy(Gx(x)) = 1 → Adversarial loss
- Gx(y) = y, Gy(x) = x → Identity loss

- **Total Loss = Adversarial loss + 𝜶 * Cycle consistency loss + 𝜷 * Identity loss**

