
import numpy as np

vector_set = []

with open('/Users/moon/Desktop/Python Practiceㅌ/plots.txt') as f: #plot.txt 파일 열기
    for line in f:
        vector_set.append(line.strip().split('\t'))
num_points= len(vector_set)
print(num_points)

# numpy array 로 변환하기
#[x1, x2] [y] 처럼
x_data = np.array([ [float(v[0]), float(v[1]),1.0] ##b부분은 1.0으로
                    for v in vector_set])
y_data = np.array([ [float(v[2])] for v in vector_set])

#처음에 W 를 임의의 값으로 정해준다.
W = np.random.randn(3,1)
for step in range(1000):
#행렬 곱으로 H = W1*x1+W2*x2+b 한꺼번에 해준다.
#cost 함수는 (H(x)-y)^2들의 평균 = loss로 표현
    H = np.dot(x_data,W)
    loss = np.sum(np.power(H- y_data,2))/num_points

#gradient 구하기 = 코스트를 W1, W2, b로 편미분한걸 W1, W2,b에서 빼주면서 맞는 W값을 찾아가는것.
#전치행렬과 H-y를 곱해주면 된다.
    z = np.dot(x_data.T, (H-y_data))/num_points
    W = W-0.05*z # 0.05씩 곱하면서 줄어들자!(임의로 정한거)
    if step&100==0:
        print(step, W.reshape(-1,3), loss)
