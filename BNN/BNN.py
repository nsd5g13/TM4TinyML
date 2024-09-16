# python==3.7.13, tensorflow==2.0.0

import tensorflow as tf
import larq as lq
import sys, os
import numpy as np

# --------------- load dataset ------------------------------------------------------------
dataset = sys.argv[1]

X_train = np.load(r'../Booleanization/bool_datasets/'+dataset+'/X_train.npy')
Y_train = np.load(r'../Booleanization/bool_datasets/'+dataset+'/Y_train.npy')
X_test = np.load(r'../Booleanization/bool_datasets/'+dataset+'/X_test.npy')
Y_test = np.load(r'../Booleanization/bool_datasets/'+dataset+'/Y_test.npy')
no_classes = len(list(set(Y_train.tolist())))

# -------------- model structure ---------------------------------------------------------
model = tf.keras.models.Sequential()

# In the first layer we only quantize the weights and not the input
model.add(lq.layers.QuantDense(512, use_bias=False, kernel_quantizer="ste_sign", kernel_constraint="weight_clip"))

model.add(lq.layers.QuantDense(no_classes, use_bias=False, input_quantizer="ste_sign", kernel_quantizer="ste_sign", kernel_constraint="weight_clip"))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Activation("softmax"))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# -------------- training and testing ----------------------------------------------------
no_epochs = 100
model.fit(X_train, Y_train, epochs=no_epochs)
test_loss, test_acc = model.evaluate(X_test, Y_test)
print(f"Test accuracy {test_acc * 100:.2f} %")

# ------------- prediction for given samples ------------------------------------------
predictions = model.predict(X_test)
predict_labels = []
for predict in predictions:
	predict_labels.append(np.argmax(predict))
print("Actual label: %d" %Y_test[1])
print("Predicted label: %d" %predict_labels[1])

# ------------ Capture hidden layer outputs -----------------------------------------
'''
new_model = tf.keras.models.Sequential()
new_model.add(lq.layers.QuantDense(128, use_bias=False, kernel_quantizer="ste_sign", kernel_constraint="weight_clip", input_dim=160))
new_model.set_weights(model.layers[0].get_weights())
new_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
output = new_model.predict(X_test)
'''

# ------------ export test samples and labels in text files --------------------------
if not os.path.exists(r"micropython_input"):
	os.makedirs(r"micropython_input")
np.savetxt(r"micropython_input/X.txt", X_test, fmt='%d', delimiter='')
f = open(r"micropython_input/Y.txt", 'w')
f.write('predict\tactual\n')
for predict, actual in zip(predict_labels, Y_test):
	f.write('%d\t%d\n' %(predict, actual))
f.close()

# ------------ export weights in a text file --------------------------------------------
weights = model.get_weights()		# These are the full-precision weights

weights0 = weights[0]
weights0 = np.transpose(weights0)

weights0[weights0 >= 0] = 1		# 1/0 indicates +1/-1
weights0[weights0 < 0] = 0
np.savetxt(r"micropython_input/weights0.txt", weights0, fmt='%d', delimiter='')

weights1 = weights[1]
weights1 = np.transpose(weights1)
weights1[weights1 >= 0] = 1		# 1/0 indicates +1/-1
weights1[weights1 < 0] = 0
np.savetxt(r"micropython_input/weights1.txt", weights1, fmt='%d', delimiter='')

lq.models.summary(model)