# Don't Merge In Any Case!!!!!!
그것만은 하지마.
---
# path.py -> dataset 경로 변경 필요
# slurm batch file 재작성 필요
# main.py에 model 불러오는 코드 작성 필요
# util 은 건들 x , optimtarget.py는 hyperparameter , dataaug.py는 건들 o

---

그냥 model 마다 브랜치 파서 기록할 것

같은 모델인 경우일 때만, optimizer, hyperparameter만 달리 했을 때 ... 최적의 acc에 대해 merge 수행할 것. 

readme에는 acc 기록
