import os, random, warnings, time
import numpy as np
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from IPython.display import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go

print ("Random number with seed 73")
random.seed(73)

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint

#from tqdm import tqdm
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt

sess=tf.compat.v1.Session()
#====================================================================
def h_plot(history):
    # plot loss during training
    fig = plt.figure(figsize=(18,10))
    plt.subplot(211)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    # plot mse during training
    plt.subplot(212)
    plt.title('Mean Squared Error')
    plt.plot(history.history['mse'], label='train')
    plt.plot(history.history['val_mse'], label='val')
    plt.legend()
    plt.show()

def q_data_split(rs_df, p_tr_sel = True):
	if p_tr_sel == True:
		Xs = rs_df[rs_df.columns[:-1]]
	elif p_tr_sel == False:
		Xs = rs_df[rs_df.columns[:-9]]
	Ys = rs_df[rs_df.columns[-1]]
	#Se calcula la cantidad de datos para train, val y test, siguiendo regla 70/20/10 respectivamente
	trn_data_count = int(len(rs_df)*0.7)
	val_data_count = int(len(rs_df)*0.2)
	tst_data_count = int(len(rs_df)*0.1)
	#===================================================================================
	x_train = Xs[:trn_data_count].to_numpy().astype('float32')
	y_train = Ys[:trn_data_count].to_numpy().astype('float32')

	x_val = Xs[trn_data_count:trn_data_count+val_data_count].to_numpy().astype('float32')
	y_val = Ys[trn_data_count:trn_data_count+val_data_count].to_numpy().astype('float32')

	x_tst = Xs[trn_data_count+val_data_count:].to_numpy().astype('float32')
	y_tst = Ys[trn_data_count+val_data_count:].to_numpy().astype('float32')

	return x_train, y_train, x_val, y_val, x_tst, y_tst

def qml_model1(dense_1 = 64, dense_2 = 64, i_shape=(16,), opt_sel='adam', l_rate=1e-3, act_in=None, act_out=None):
	model = None
	del model
	#tf.keras.backend.clear_session()
	tf.compat.v1.keras.backend.clear_session()
	tf.compat.v1.keras.backend.set_session(sess)
	#K.clear_session()
	#K.set_session(sess)
	#=========================================
	if opt_sel == 'adam':
		opt = tf.keras.optimizers.Adam(learning_rate=l_rate)
	elif opt_sel == 'sgd':
		opt = tf.keras.optimizers.SGD(learning_rate=l_rate)
	elif opt_sel == 'adadelta':
		opt = tf.keras.optimizers.Adadelta(learning_rate=l_rate)
	elif opt_sel == 'adagrad':
		opt = tf.keras.optimizers.Adagrad(learning_rate=l_rate)
	elif opt_sel == 'adamax':
		opt = tf.keras.optimizers.Adamax(learning_rate=l_rate)
	elif opt_sel == 'nadam':
		opt = tf.keras.optimizers.Nadam(learning_rate=l_rate)
	#=========================================
	b_name = 'qml_model_param_exploration'
	inputs = Input(shape=i_shape, name='main_input')
	main_branch = Dense(24, kernel_initializer = 'normal', activation = act_in, name='dense_0_'+b_name)(inputs)
	main_branch = Dense(32, kernel_initializer = 'normal', activation = act_in, name='dense_1_'+b_name)(main_branch)
	main_branch = Dense(64, kernel_initializer = 'normal', activation = act_in, name='dense_2_'+b_name)(main_branch)
	main_branch = Dense(64, kernel_initializer = 'normal', activation = act_in, name='dense_3_'+b_name)(main_branch)
	#====================================================
	main_branch = Dropout(0.2, name='drop_out_1_'+b_name)(main_branch)
	base_branch_out = Dense(1, activation = act_out, name='out_'+b_name)(main_branch)

	model = Model(inputs = inputs, outputs = base_branch_out)
	model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
	return model

def qml_model2(dense_1 = 64, dense_2 = 64, i_shape=(16,), opt_sel='adam', l_rate=1e-3, act_in=None, act_out=None):
	model = None
	del model
	# cleanup
	#tf.keras.backend.clear_session()
	tf.compat.v1.keras.backend.clear_session()
	tf.compat.v1.keras.backend.set_session(sess)
	#K.clear_session()
	#K.set_session(sess)
	#=========================================
	if opt_sel == 'adam':
		opt = tf.keras.optimizers.Adam(learning_rate=l_rate)
	elif opt_sel == 'sgd':
		opt = tf.keras.optimizers.SGD(learning_rate=l_rate)
	elif opt_sel == 'adadelta':
		opt = tf.keras.optimizers.Adadelta(learning_rate=l_rate)
	elif opt_sel == 'adagrad':
		opt = tf.keras.optimizers.Adagrad(learning_rate=l_rate)
	elif opt_sel == 'adamax':
		opt = tf.keras.optimizers.Adamax(learning_rate=l_rate)
	elif opt_sel == 'nadam':
		opt = tf.keras.optimizers.Nadam(learning_rate=l_rate)
	#=========================================
	b_name = 'qml_model_param_exploration'
	inputs = Input(shape=i_shape, name='main_input')
	main_branch = Dense(24, kernel_initializer = 'normal', activation = act_in, name='dense_0_'+b_name)(inputs)
	main_branch = Dense(64, kernel_initializer = 'normal', activation = act_in, name='dense_1_'+b_name)(main_branch)
	main_branch = Dense(128, kernel_initializer = 'normal', activation = act_in, name='dense_2_'+b_name)(main_branch)
	main_branch = Dense(128, kernel_initializer = 'normal', activation = act_in, name='dense_3_'+b_name)(main_branch)
	#====================================================
	main_branch = Dropout(0.2, name='drop_out_1_'+b_name)(main_branch)
	base_branch_out = Dense(1, activation = act_out, name='out_'+b_name)(main_branch)

	model = Model(inputs = inputs, outputs = base_branch_out)
	model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
	return model

def qml_model3(l_rate=1e-3, dense_1 = 64, dense_2 = 64, i_shape=(16,), opt_sel='adam', act_in=None, act_out=None):
	model = None
	del model
	# cleanup
	#tf.keras.backend.clear_session()
	tf.compat.v1.keras.backend.clear_session()
	tf.compat.v1.keras.backend.set_session(sess)
	#K.clear_session()
	#K.set_session(sess)
	#=========================================
	if opt_sel == 'adam':
		opt = tf.keras.optimizers.Adam(learning_rate=l_rate)
	elif opt_sel == 'sgd':
		opt = tf.keras.optimizers.SGD(learning_rate=l_rate)
	elif opt_sel == 'adadelta':
		opt = tf.keras.optimizers.Adadelta(learning_rate=l_rate)
	elif opt_sel == 'adagrad':
		opt = tf.keras.optimizers.Adagrad(learning_rate=l_rate)
	elif opt_sel == 'adamax':
		opt = tf.keras.optimizers.Adamax(learning_rate=l_rate)
	elif opt_sel == 'nadam':
		opt = tf.keras.optimizers.Nadam(learning_rate=l_rate)
	#=========================================
	b_name = 'qml_model_param_exploration'
	inputs = Input(shape=i_shape, name='main_input')
	main_branch = Dense(24, kernel_initializer = 'normal', activation = act_in, name='dense_0_'+b_name)(inputs)
	main_branch = Dense(64, kernel_initializer = 'normal', activation = act_in, name='dense_1_'+b_name)(main_branch)
	main_branch = Dense(128, kernel_initializer = 'normal', activation = act_in, name='dense_2_'+b_name)(main_branch)
	main_branch = Dense(64, kernel_initializer = 'normal', activation = act_in, name='dense_3_'+b_name)(main_branch)
	#====================================================
	main_branch = Dropout(0.2, name='drop_out_1_'+b_name)(main_branch)
	base_branch_out = Dense(1, activation = act_out, name='out_'+b_name)(main_branch)

	model = Model(inputs = inputs, outputs = base_branch_out)
	model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
	return model

def train_explore_model(exp_name=None, n_epochs=5, model_sel=None, in_dim_s = None, in_act = None, out_act = None,
	x_train=None, y_train=None, x_val=None, y_val=None, x_tst=None, y_tst=None):
	#Results dataframe
	col_names = ['exp_name', 'exp_number','optimizer', 'b_s', 'lr', 'trn_loss', 'trn_mse', 'val_loss', 'val_mse', 'tst_mse']
	results_df = pd.DataFrame(columns = col_names)
	#==============================================
	#opt_list = ['adam', 'sgd', 'adadelta', 'adagrad', 'adamax', 'nadam']
	opt_list = ['adam', 'adamax', 'nadam']
	#batches = [2, 4, 8, 16, 32]
	batches = [2, 4, 8]
	#l_rate = [1e-3, 1e-4, 1e-5, 1e-6]
	l_rate = [1e-3, 1e-4]
	#==============================================
	experiment = 1
	
	start_total = time.time()
	for x in range(len(opt_list)):
		for y in range(len(batches)):
			for z in range(len(l_rate)):
				opt_name = opt_list[x]
				b_s = batches[y]
				l_r = l_rate[z]
				print('Experiment: %2d, optimizer: %s, bs: %2d, lr: %10.1E' % (experiment, opt_name, b_s, l_r) )
				#================================
				if model_sel == 1:
					model = qml_model1(i_shape=(in_dim_s,), opt_sel=opt_name, l_rate=l_r, act_in=in_act, act_out=out_act)
				elif model_sel == 2:
					model = qml_model2(i_shape=(in_dim_s,), opt_sel=opt_name, l_rate=l_r, act_in=in_act, act_out=out_act)
				elif model_sel == 3:
					model = qml_model3(i_shape=(in_dim_s,), opt_sel=opt_name, l_rate=l_r, act_in=in_act, act_out=out_act)
				#m_name = './weights_v0/2qb/' + exp_name + '_' + str(experiment) + '.h5'
				#m_name = './weights_v2/2qb/' + exp_name + '_' + str(experiment) + '.h5'#For pure states
				#m_name = './weights_v2/2qb_1M/' + exp_name + '_' + str(experiment) + '.h5'#For pure states
				#m_name = './weights_v2/2qb_10K/' + exp_name + '_' + str(experiment) + '.h5'#For pure states
				m_name = './weights_v2/2qb_1k_50AB/' + exp_name + '_' + str(experiment) + '.h5'#For mixed dataset 50A50B
				model_checkpoint = ModelCheckpoint(m_name, verbose=1, monitor='loss', save_best_only=True, mode='auto')
				#================================
				start = time.time()
				history = model.fit(x_train, y_train, validation_data=(x_val,y_val), verbose=1,
					epochs=n_epochs, batch_size=b_s, callbacks=[model_checkpoint])
				experiment+=1
				elapsed_partial = (time.time() - start)
				#================================
				print('='*20)
				print('Results')
				print('='*20)
				print('Partial elapsed training time was: ', (elapsed_partial/3600), ' hours')
				# Evaluate the model
				_, train_mse = model.evaluate(x_train, y_train, verbose=0)
				_, test_mse = model.evaluate(x_tst, y_tst, verbose=0)
				print('Eval: Train: %.4f, Test: %.4f' % (train_mse, test_mse))
				#Builind results
				min_trn_loss = min(history.history['loss'])
				min_trn_loss_idx = history.history['loss'].index(min_trn_loss)
				trn_mse = history.history['mse'][min_trn_loss_idx]
				val_loss = history.history['val_loss'][min_trn_loss_idx]
				val_mse = history.history['val_mse'][min_trn_loss_idx]

				results_df = results_df.append(pd.Series([exp_name, int(experiment), opt_name, int(b_s),
					l_r, min_trn_loss, trn_mse, val_loss, val_mse, test_mse],
					index = results_df.columns), ignore_index=True)
				h_plot(history)
				elapsed_total = (time.time() - start_total)
	print("Total elapsed training time was: %0.10f seconds" % elapsed_total)
	print('Total elapsed training time was: ', (elapsed_total/3600), ' hours')

	results_df_sort = results_df.sort_values(by='tst_mse', ascending=True)#.reset_index(drop=True, inplace=True)
	results_df_sort.reset_index(drop=True, inplace=True)
	#results_df_sort.to_csv('./results_v0/'+exp_name+'.csv', index=False, index_label=False)
	#results_df_sort.to_csv('./results_v2/'+exp_name+'.csv', index=False, index_label=False)#For pure states
	#results_df_sort.to_csv('./results_v2_1M/'+exp_name+'.csv', index=False, index_label=False)#For pure states
	#results_df_sort.to_csv('./results_v2_10K/'+exp_name+'.csv', index=False, index_label=False)#For pure states
	results_df_sort.to_csv('./results_v2_1k_50AB/'+exp_name+'.csv', index=False, index_label=False)#For mixed dataset 50A50B
	return results_df_sort

def train_best(n_epochs=5, in_dim_s = None, m2t = None,
	x_train=None, y_train=None, x_val=None, y_val=None, x_tst=None, y_tst=None):
	col_names = ['exp_name', 'exp_number','optimizer', 'b_s', 'lr', 'trn_loss', 'trn_mse', 'val_loss', 'val_mse', 'tst_mse']
	results_df = pd.DataFrame(columns = col_names)
	experiment = 1
	start_total = time.time()
	for key in m2t.keys():
		model_sel = m2t[key][0]
		exp_name = key+'_'+model_sel
		opt_name = m2t[key][1]
		b_s = m2t[key][2]
		l_r = m2t[key][3]
		in_act = m2t[key][4]
		out_act = m2t[key][5]
		print('Experiment: %2d, optimizer: %s, bs: %2d, lr: %10.1E' % (experiment, opt_name, b_s, l_r) )
		if model_sel == 'm1':
			model = qml_model1(i_shape=(in_dim_s,), opt_sel=opt_name, l_rate=l_r, act_in=in_act, act_out=out_act)
		elif model_sel == 'm2':
			model = qml_model2(i_shape=(in_dim_s,), opt_sel=opt_name, l_rate=l_r, act_in=in_act, act_out=out_act)
		elif model_sel == 'm3':
			model = qml_model3(i_shape=(in_dim_s,), opt_sel=opt_name, l_rate=l_r, act_in=in_act, act_out=out_act)

		#m_name = './weights_v0/2qb_best/' + exp_name + '.h5'
		#m_name = './weights_v2/2qb_best/' + exp_name + '.h5'#For pure states
		#m_name = './weights_v2/2qb_best_1M/' + exp_name + '.h5'#For pure states 1M
		#m_name = './weights_v2/2qb_best_1M/' + exp_name + '.h5'#For pure states 1M
		m_name = './weights_v2/2qb_best_1k_50AB/' + exp_name + '.h5'#For mixed dataset 50A50B
		model_checkpoint = ModelCheckpoint(m_name, verbose=1, monitor='loss', save_best_only=True, mode='auto')
		start = time.time()
		history = model.fit(x_train, y_train, validation_data=(x_val,y_val), verbose=1,
			epochs=n_epochs, batch_size=b_s, callbacks=[model_checkpoint])
		experiment+=1
		elapsed_partial = (time.time() - start)
		#================================
		print('='*20)
		print('Results')
		print('='*20)
		print('Partial elapsed training time was: ', (elapsed_partial/3600), ' hours')
		# Evaluate the model
		_, train_mse = model.evaluate(x_train, y_train, verbose=0)
		_, test_mse = model.evaluate(x_tst, y_tst, verbose=0)
		print('Eval: Train: %.4f, Test: %.4f' % (train_mse, test_mse))
		#Builind results
		min_trn_loss = min(history.history['loss'])
		min_trn_loss_idx = history.history['loss'].index(min_trn_loss)
		trn_mse = history.history['mse'][min_trn_loss_idx]
		val_loss = history.history['val_loss'][min_trn_loss_idx]
		val_mse = history.history['val_mse'][min_trn_loss_idx]

		results_df = results_df.append(pd.Series([exp_name, int(experiment), opt_name, int(b_s),
			l_r, min_trn_loss, trn_mse, val_loss, val_mse, test_mse],
			index = results_df.columns), ignore_index=True)
		h_plot(history)
		elapsed_total = (time.time() - start_total)

	print("Total elapsed training time was: %0.10f seconds" % elapsed_total)
	print('Total elapsed training time was: ', (elapsed_total/3600), ' hours')

	results_df_sort = results_df.sort_values(by='tst_mse', ascending=True)#.reset_index(drop=True, inplace=True)
	results_df_sort.reset_index(drop=True, inplace=True)
	#results_df_sort.to_csv('./results_v0/'+'best_exp4'+'.csv', index=False, index_label=False)
	#results_df_sort.to_csv('./results_v2/'+'best_exp4'+'.csv', index=False, index_label=False)#For pure states
	results_df_sort.to_csv('./results_v2/'+'best_exp4_50AB'+'.csv', index=False, index_label=False)#For mixed dataset 50A50B
	return results_df_sort