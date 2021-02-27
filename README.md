# TensorFlow2.0  代码摘于吴恩达老师云课堂中的原代码是用tensorflow1.0所写，下文用2.0 版本
Build a Simple TensorFlow Example Use TensorFlow2.0 

import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
#tf.compat.v1.disable_eager_execution()
tf.disable_v2_behavior()


coefficigents=np.array([[1.],[-20],[100.]])
w=tf.Variable(0,dtype=tf.float32)
x=tf.placeholder(tf.float32,[3,1])
#cost=tf.add(tf.add(w**2,tf.multiply(-10.,w)),25)
#cost=w**2-10*w+25
cost=x[0][0]*w**2+x[1][0]*w+x[2][0]
train=tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init=tf.global_variables_initializer()
session=tf.Session()
session.run(init)
print(session.run(w))


session.run(train,feed_dict={x:coefficigents})
print(session.run(w))


for i in range(1000):
    session.run(train,feed_dict={x:coefficigents})
print(session.run(w))
