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
  
## 🥬 Prediction of bok choy growth 🥬
![image](https://user-images.githubusercontent.com/72390138/205493584-af95700c-c420-4f95-a5fc-d1c05bb27bc7.png)      

dacon AI 경진대회 : **청경채 성장률 예측하기**                    
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
  
  
### 🍀 Submission & Score
📌Tensorflow      
public : 17.91, private : 17.53     
📌 Pytorch    
public : 19.2138, private : 18.009    
📌LSTM      
public : 21.5578 / private 22.5213      
