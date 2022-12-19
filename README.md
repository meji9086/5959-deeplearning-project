# 59️⃣59️⃣-deeplearning-project        
**🦁 Likelion AI SCHOOL7 👶🤱 으샤으샤팀3 9조 오9️⃣오9️⃣ 팀**    


#### 👨‍👩‍👧‍👦 Team Info.    
|name|github|velog&blog|                       
|:---:|:---:|:---:|                                        
|<span style="color:blue">[김예지👑](https://github.com/meji9086)</span>|<a href="https://github.com/meji9086"><img src="https://img.shields.io/badge/github-181717?style=flat-square&logo=github&logoColor=White"/>|<a href="https://blog.naver.com/meji9086"><img src="https://img.shields.io/badge/blog-09B3AF?style=flat-square&logo=Storyblok&logoColor=white"/>|                                    
|<span style="color:blue">[이정은](https://github.com/LJEDD2)</span>|<a href="https://github.com/hooun"><img src="https://img.shields.io/badge/github-181717?style=flat-square&logo=github&logoColor=White"/>|<a href="https://blog.naver.com/charzim0611"><img src="https://img.shields.io/badge/blog-09B3AF?style=flat-square&logo=Storyblok&logoColor=white"/>|                  
|<span style="color:blue">[조예슬](https://github.com/seul1230)</span>|<a href="https://github.com/tvn123"><img src="https://img.shields.io/badge/github-181717?style=flat-square&logo=github&logoColor=White"/>|<a href="https://seul1230.github.io/"><img src="https://img.shields.io/badge/blog-09B3AF?style=flat-square&logo=Storyblok&logoColor=white"/>|                    
|<span style="color:blue">[임종우](https://github.com/imngooh)</span>|<a href="https://github.com/dkssudgb"><img src="https://img.shields.io/badge/github-181717?style=flat-square&logo=github&logoColor=White"/>|<a href="https://velog.io/@im_ngooh"><img src="https://img.shields.io/badge/velog-20C997?style=flat-square&logo=velog&logoColor=white"/>|            
|<span style="color:blue">[권태윤](https://github.com/taeyoon94)</span>|<a href="https://github.com/taeyoon94"><img src="https://img.shields.io/badge/github-181717?style=flat-square&logo=github&logoColor=White"/>|<a href="https://github.com/taeyoon94"><img src="https://img.shields.io/badge/github-181717?style=flat-square&logo=github&logoColor=White"/>          

#### 💡 Project Info.         
|number|title|link|bestscore|
|:-:|:----:|:----:|:-:|         
|1|🥬 Prediction of bok choy growth|[https://dacon.io/competitions/official/235961/overview/description](https://dacon.io/competitions/official/235961/overview/description)|private mse 17.53|         
|2|😷 Face Mask Classification|[https://www.kaggle.com/datasets/dhruvmak/face-mask-detection](https://www.kaggle.com/datasets/dhruvmak/face-mask-detection)|accuracy 0.97|        
|3|📝 Sentences type Classification|[https://dacon.io/competitions/official/236037/overview/description](https://dacon.io/competitions/official/236037/overview/description)|private accuracy 0.7559|        

## 1. 🥬 Prediction of bok choy growth 🥬
![image](https://user-images.githubusercontent.com/72390138/205493584-af95700c-c420-4f95-a5fc-d1c05bb27bc7.png)      

🏆 dacon AI 경진대회 : **청경채 성장률 예측하기**                    
주소 : [https://dacon.io/competitions/official/235961/overview/description](https://dacon.io/competitions/official/235961/overview/description)        
  
📜 notion : https://www.notion.so/MINI5-AI-b031d68247e24a30b192b24c522284d1
  
### 📃 summary     
<img src="https://user-images.githubusercontent.com/72390138/205493735-e21c9908-44e8-45e9-8a9c-f3942f203fa7.png" weight="350" height="350">        
4차 산업혁명 시대를 맞아 농업 분야에서도 AI 기술이 널리 사용되어 IT 기술을 동원한 스마트팜 등 더욱 효율적인 작물 재배가 가능해지고 있습니다. 작물의 효율적인 생육을 위한 최적의 환경을 도출한다면 식물 재배에 큰 도움이 될 것이며, 청경채 뿐만 아닌 모든 작물 재배율이 좋아질 것입니다. 미래의 작물 재배에서는 이 데이터를 가지고 인공지능을 이용하여 작물별 맞춤형 솔루션을 농업인들이 편리하고 친근하게 생활 속에서 활용하는 첫 걸음을 내딛을 수 있을 것입니다.      
 
  
따라서, **인공지능(AI)을 활용하여 국내 고유 식물 자원에서 유용한 천연물 소재를 탐색하고, 그 효능과 활성 등에 대해 연구하는 것이 목표**입니다.     
실제 AI를 이용한 작물을 재배하는 스마트팜과 같은 곳에 유용하게 사용될 것입니다.      
  
  
### 🗂 Data info.     
**dacon 청경채 예측 데이터** : [https://dacon.io/competitions/official/235961/data](https://dacon.io/competitions/official/235961/data)   

**📁 train input dataset[folder]**       
![image](https://user-images.githubusercontent.com/72390138/205494867-68e0c49a-3740-4fe4-8812-da281ace0524.png)    
총 58개 청경채 케이스를 각 청경채 케이스 별 환경 데이터(1분 간격)으로 구성되어 있음      

**📁 train target dataset[folder]**                  
<img src="https://user-images.githubusercontent.com/72390138/205494955-e4752ba6-3a90-41de-a1ee-ec2929ea8dd6.png" weight="300" height="500">     
총 58개 청경채 케이스를 rate column의 각 청경채 케이스 별 잎 면적 증감률(1일 간격)로 구성되어 있음       

**📂 train(input+target) shape**                   
train(input+target) (1813, 43)         
test(input+target) (195, 43)            
  
  
### 📊 Visualization        
1️⃣ 내부온도관측치, 내부습도관측치, 총추정광량, 월별 rate        
![image](https://user-images.githubusercontent.com/72390138/205497279-d59cb01a-a52e-44f7-9cef-c34381f8fcfd.png)              

2️⃣ 적색, 청색, 백색, 총추 추정광량 별 rate
![image](https://user-images.githubusercontent.com/72390138/205497255-0030b727-3f14-4c07-b22b-806a05334616.png)             
백색과 총추는 100에서, 적색과 청색은 0에서 성장률이 높다.     
  
3️⃣ EC와 CO2의 냉방상태       
![image](https://user-images.githubusercontent.com/72390138/205496256-fe545a8c-16b8-4644-a339-030a2d05f271.png)       
EC관측치가 클수록 냉방상태는 적었으며, 반대로 작을수록 냉방상태는 높은 것을 확인할 수 있다.   

4️⃣ 각 CASE 별 잎면적 증감률(rate) 변화      
![image](https://user-images.githubusercontent.com/72390138/205496376-956636fd-c78d-433e-b690-bdb8764cfe71.png)      
분포를 일정하게 만들기 위한 Scaling 과정이 필요함이 보였다.          
  
 
### 🔍 Modeling
📌 Scaling - RobustScaler      
```python      
from sklearn.preprocessing import RobustScaler        
rb = RobustScaler()         
train_X = rb.fit_transform(train_X)         
test_X = rb.transform(test_X)            
```
  
📌 Tensorflow    
```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=128, input_shape=[input_shape]),
    tf.keras.layers.Dense(128, activation='selu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1)
])
  
optimizer = tf.keras.optimizers.RMSprop(0.001)     
  
model.compile(optimizer=optimizer, 
              loss=["mae", "mse"], 
              metrics=["mae", "mse"])
```

📌 Pytorch     
```python      
linear1 = torch.nn.Linear(train_X.shape[1], 512, bias=True)
linear3 = torch.nn.Linear(512, 256, bias=True)
linear4 = torch.nn.Linear(256, 128, bias=True)
linear5 = torch.nn.Linear(128, 64, bias=True)
linear6 = torch.nn.Linear(64, 32, bias=True)
linear7 = torch.nn.Linear(32, 10, bias=True)
linear8 = torch.nn.Linear(10, 1, bias=True)

relu = torch.nn.ReLU()
dropout = torch.nn.Dropout(p=0.1)
  
model = torch.nn.Sequential(linear1,relu,
                            linear3,relu,
                            linear4, relu,
                            linear5, relu,
                            linear6, relu,
                            linear7, relu,
                            linear8).to(device)

# nn 패키지를 사용하여 모델과 손실 함수를 정의합니다.
loss_fn = torch.nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 가중치 초기화 
torch.nn.init.xavier_normal_(linear1.weight)
torch.nn.init.xavier_normal_(linear2.weight)
torch.nn.init.xavier_normal_(linear3.weight)
torch.nn.init.xavier_normal_(linear4.weight)
torch.nn.init.xavier_normal_(linear5.weight)
torch.nn.init.xavier_normal_(linear6.weight)
torch.nn.init.xavier_normal_(linear7.weight)
torch.nn.init.xavier_normal_(linear8.weight)     
  
Parameter containing:
tensor([[-0.1239,  0.3789, -0.2748,  0.2951,  0.0612,  0.2898,  0.2660, -0.4028,
         -1.1738,  0.6484]], requires_grad=True)        
```

📌 LSTM    
```python    
class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__() 
        self.lstm = nn.LSTM(input_size=train_X.shape[1], hidden_size=256, batch_first=True, bidirectional=False) # LSTM 메모리 추가 
        self.dense = nn.Sequential(
            # nn.Linear(256, 10 , torch.nn.ReLU()),
            # Layer 추가 
            nn.Linear(256, 128,  torch.nn.ReLU()),
            nn.Linear(128, 64, torch.nn.ReLU()),
            nn.Linear(64, 32,  torch.nn.ReLU()),
            nn.Linear(32, 10,  torch.nn.ReLU()),
            nn.Linear(10, 1)
        )
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        hidden, _ = self.lstm(x)
        x = self.dropout(hidden)
        output = self.dense(hidden)
        return output
  
model = BaseModel()
model.eval() 
loss_fn = torch.nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)    
```
  
  
### 🍀 Submission & Score (mse)
📌Tensorflow      
public : 17.91, private : 17.53     
📌 Pytorch    
public : 19.2138, private : 18.009    
📌LSTM      
public : 21.5578 / private 22.5213      

---

## 2. 😷 Face Mask Classification 😷     
<img src="https://user-images.githubusercontent.com/72390138/206939017-352d4a33-79df-4d47-873c-4fefd40c5884.png" weight="350" height="190">        

🏆 Kaggle AI 경진대회 : **마스크 착용/미착용 분류**                    
주소 : [https://www.kaggle.com/datasets/dhruvmak/face-mask-detection](https://www.kaggle.com/datasets/dhruvmak/face-mask-detection)        
  
📜 notion : https://www.notion.so/MINI6-Mask_or_No_Mask-Classification-a7d66cebd161444180e9024e13be2f98#35d0f4877a1f4aa3bd9a0719fc5bea2d     
📌 streamlit : https://seul1230-mask-classification-main-pqg0f0.streamlit.app/
📌 streamlit : https://imngooh-mini-streamlit-app-82wtn0.streamlit.app/
  
### 📃 summary     
<img src="https://user-images.githubusercontent.com/72390138/206940326-e4e4d101-7939-4072-8c9c-65f5e03734fa.png" weight="400" height="400">         
코로나19 바이러스로 인한 마스크 착용 의무화하였었고, 그에 따른 마스크 미착용자에 대한 과태로 부과 대상에 처했었다.     
마스크 착용과 미착용의 분류를 통해 모니터링하는 인력을 감소화하고 마스크 착용의 의무화를 느끼고 착용률을 높이고자 한다.    

### 🗂 Data info.  
**kaggle 마스크 착용 여부 이미지 데이터** : [https://www.kaggle.com/datasets/dhruvmak/face-mask-detection](https://www.kaggle.com/datasets/dhruvmak/face-mask-detection)   

**📁 with mask[folder]**       
<img src="https://user-images.githubusercontent.com/72390138/206941647-f46f60a9-3f55-492c-8430-a7e0bce5b8b6.png" weight="500" height="500">         
총 220개의 마스크 착용한 사람들의 이미지

**📁 without mask[folder]**                  
<img src="https://user-images.githubusercontent.com/72390138/206941671-6211b537-b270-45d3-8fc0-c0d6f1f7285f.png" weight="500" height="500">       
총 220개의 마스크 미착용한 사람들의 이미지      

**📂 train/valid/test shape**                    
train_df (281, 2)          
val_df (71, 2)       
test_df (88, 2)        

### 📊 Visualization    
✅ Target Ratio     
<img src="https://user-images.githubusercontent.com/72390138/206941876-1053e395-212a-43ff-b40f-f54566dab42f.png" weight="350" height="350">         

### 🔍 Modeling         
**⭐ Tensorflow를 이용한 모델링 ⭐**                
📌 Resnet152V       
```python
# imagenet으로 pre-trained 된 가중치 값 적용
md = ResNet152V2(include_top=False, pooling='max', 
                  weights='imagenet', input_shape=(height, width, 3))
md.trainable=False

model = models.Sequential()
model.add(md)
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])
              
early_stop = EarlyStopping(patience=5)

history = model.fit(train_datagen, epochs=20, 
                    validation_data=val_datagen,
                    validation_steps=len(val_datagen),
                    callbacks = [early_stop])         
```   
<img src="https://user-images.githubusercontent.com/72390138/206942373-bc752360-e652-4c91-be5f-ba8d5addb118.png" weight="250" height="250">         


📌 VGG19      
```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.applications import vgg19

vgg = vgg19.VGG19(
    include_top = False,
    weights = 'imagenet',
    input_shape = (height, width, 3)
)

model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))

model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor = 'val_loss', patience = 10)

history = model.fit(
    train_dataset,
    epochs = 100,
    validation_data = valid_dataset,
    callbacks = [early_stop]
```        
<img src="https://user-images.githubusercontent.com/72390138/206942406-f0be43cf-3d94-42f6-9991-9a2c7fcd725c.png" weight="350" height="350">         
  
📌 DenseNet121       
```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.applications import densenet

densenet = densenet.DenseNet121(
    include_top = False,
    weights = 'imagenet',
    input_shape = (height, width, 3),
    pooling = 'avg'
)

modeld = Sequential()
modeld.add(densenet)
modeld.add(Flatten())
modeld.add(Dense(1, activation = 'sigmoid'))

modeld.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor = 'val_loss', patience = 10)

history = modeld.fit(
    train_dataset,
    epochs = 100,
    validation_data = valid_dataset,
    callbacks = [early_stop]
)
```    
<img src="https://user-images.githubusercontent.com/72390138/206942427-2138ee68-d95b-4edc-8abb-39380b53a913.png" weight="350" height="350">          
  
### 🍀 Submission & Score
📌 Resnet152V  -> Best Score     
<img src="https://user-images.githubusercontent.com/72390138/206942588-a344bb5b-e72e-4909-b12d-777fbd5a6906.png" weight="350" height="450">         

📌 팀원들의 이미지를 이용한 마스크 착용/미착용 예측
<img src="https://user-images.githubusercontent.com/72390138/206942609-2b1b424c-a72c-498c-ae7e-71da416d8ec6.png" weight="500" height="550">         

---

## 3. 📝 Sentences type Classification 📝        
![image](https://user-images.githubusercontent.com/72390138/208331196-715359e9-52b9-43cf-b5e9-4c6793a98c16.png)    

🏆 dacon AI 경진대회 : **문장 유형 분류하기**                    
주소 : [https://dacon.io/competitions/official/236037/overview/description](https://dacon.io/competitions/official/236037/overview/description)        
  
📜 notion : https://www.notion.so/Dacon-AI-54fe914d79584be2a7bd9404ee6fa9a5
  
### 📃 summary     
<img src="https://user-images.githubusercontent.com/72390138/208331552-42328ba1-475f-4332-9fe5-74658283ce4f.png" weight="400" height="400">               
하루 종일 한 마디도 하지 않는 날은 상상할 수 있지만, 한 글자도 읽지 않는 날은 상상할 수 없는 것처럼 우리는 수많은 문장과 글 속에 둘러싸여 살고 있다.       
특히나 코로나19 이후, 우리 사회에선 온라인과 비대면 소통이 주된 교류 방식이 되었다. 그만큼 우리는 이전보다도 더 많은 문장을 읽고 쓰며 세상과 소통하고 있다.           
이처럼 수많은 글들을 AI 모델을 활용해 학습하고, 빠르게 분류할 수 있다면 우리는 더 정교하게 분류된 정보를 얻고, 이를 통해 언어가 쓰이는 모든 영역에서 보다 사용자 친화적인 서비스를 경험할 수 있게 될 것이다.      


따라서, 한 발 더 나아가 한국어 인공지능 기술 고도화의 발판을 마련할 수 있도록 창의적인 문장 유형 분류 AI 모델을 만드는 것이 목표이다.     

### 🗂 Data info.     
**dacon 청경채 예측 데이터** : [https://dacon.io/competitions/official/236037/data](https://dacon.io/competitions/official/236037/data)    

**📁 train dataset**       
<img src="https://user-images.githubusercontent.com/72390138/208332994-75fab7bd-0a1c-48c2-9626-5de87625e60b.png" weight="400" height="400">          
문장에 따른 유형, 극성, 시제, 확실성이 구분되어 있음 (총 72개 종류의 Class 존재)           

**📁 train dataset**     
<img src="https://user-images.githubusercontent.com/72390138/208333183-925307ba-d173-4f4f-a62b-5a2f0d6becac.png" weight="300" height="350">       
총 7090 문장으로 구성되어 있음        

**📂 train/test shape**                   
train (16541, 7)         
test (7090, 2)             

### 📊 Visualization        
**1️⃣ 각 label의 빈도수 시각화**                 
<img src="https://user-images.githubusercontent.com/72390138/208333478-daf842be-bde7-4413-be56-fe68cbf5ef59.png" weight="400" height="400">            
극성, 유형, 확실성은 한 곳으로 치우쳐진 label이 있다.     
이에 따른 해결책을 생각해야 할 것이다.      

**2️⃣ 하나로 통합한 label의 빈도수 시각화**         
<img src="https://user-images.githubusercontent.com/72390138/208333756-bfa89cc9-b8bd-41ea-90c8-2b2bbcc4d2f6.png" weight="400" height="400">            
각 클래스 사이에서 불균형이 일어남을 알 수 있다.      

### 🔍 Modeling         
**📌 Tensorflow**    
```python
early_stop = EarlyStopping(monitor='val_loss', patience=5)
embedding_dim = 256

def Model(label):
    n_class = train_labels[label].shape[1]
    
    # 모델 정의
    model = Sequential()
    model.add(Embedding(input_dim = 77426,
                        output_dim = embedding_dim,
                        input_length = 80))
   
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(LSTM(64))
    
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(n_class, activation = 'softmax'))

    model.compile(optimizer = 'adam',
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    
    # 모델 서머리
    display(model.summary())

    # 모델 피팅
    history = model.fit(train_vec, train_labels[label], epochs = 100, validation_data = (val_vec, val_labels[label]), callbacks = [early_stop])
    df_hist = pd.DataFrame(history.history)

    print('*'*5, '학습완료', '*'*5)
    df_hist[['loss','val_loss']].plot()
    df_hist[['accuracy','val_accuracy']].plot()
    
    # 모델 평가
    print('valid 평가 결과')
    model.evaluate(val_vec, val_labels[label])

    # 예측
    y_pred = model.predict(test_vec)
    y_predict = np.argmax(y_pred, axis = 1)
    return y_predict
```

**📌 pytorch**    
```python
class BaseModel(nn.Module):
    def __init__(self, input_dim=9351):
        super(BaseModel, self).__init__()
        self.feature_extract = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
        )
        self.type_classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=512, out_features=4),
        )
        self.polarity_classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=512, out_features=3),
        )
        self.tense_classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=512, out_features=3),
        )
        self.certainty_classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=512, out_features=2),
        )
            
    def forward(self, x):
        x = self.feature_extract(x)
        # 문장 유형, 극성, 시제, 확실성을 각각 분류
        type_output = self.type_classifier(x)
        polarity_output = self.polarity_classifier(x)
        tense_output = self.tense_classifier(x)
        certainty_output = self.certainty_classifier(x)
        return type_output, polarity_output, tense_output, certainty_output
```   

### 🍀 Submission & Score      
📌 Tensorflow : 0.5794        
📌 pytorch(baseline) : 0.5362        
📌 best score : 0.7559 (대회가 끝나지 않아 모델은 미공개)           

---
