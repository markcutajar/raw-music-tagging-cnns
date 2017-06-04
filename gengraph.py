#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import pydst.nn_utils

import scipy.io.wavefile as wavf


class tfgraph(object):

	def __init__(self, graph):
		self.graph=graph
		self.layer_outs = []
	
	
	def add_ffl(self, inputs, outsize, nonlin=tf.nn.relu, batchNorm=False, keepProb):
		with self.graph:
			output = ff_layer(inputs, outsize, nonlin, batchNorm, keepProb)	
		self.layer_outs.append(output)

	
	def add_conv1l(self, inputs, outsize, kersize, nonlin=tf.nn.relu, batchNorm=False, keepProb=1, padding, dilution):
		with self.graph:
			output = conv1_layer(inputs, outsize, kersize, nonlin, stride,
					batchNorm, keepProb, padding, dilution)
		self.layer_outs.append(output)


	def add_conv2l(self, inputs, outsize, kersize, nonlin=tf.nn.relu, batchNorm=False, keepProb=1, padding, dilution):
		with self.graph:
			output = conv2_layer(inputs, outsize, kersize, nonlin, stride,
					batchNorm, keepProb, padding, dilution)
		self.layer_outs.append(output)


	def add_maxpool(self, inputs, div, padding="SAME"):
		with self.graph:
			output = maxpool(inputs, div, padding)
		self.layer_outs.append(output)


	def add_avgpool(self, inputs, div, padding="SAME"):	
		with self.graph:
			output = avgpool(inputs, div, padding)
		self.layer_outs.append(output)


	def add_batchnorm(self):
		with self.graph:
			output = batchNorm(inputs):
		self.layer_outs.append(output)


	def initialise(self):
		with self.graph:
			# initialise
	
	
	def add_summary(self, name, value, type):
		
		assert type in ['scalar', 'histogram', 'image'], (
			'Expected type to be either scalar, histogram or image. '
            		'Got {0}'.format(type)
        	)

		with self.graph:
			if type='scalar:
				tf.summary.scalar(name, value)
			elif type='histogram':
				tf.summary.histogram(name, value)
			else: 
				tf.summary.image(name, value)

	
	def merge_summaries(self):
		with self.graph:
			merged = tf.summary.merge_all()
			return merged

	def train(self, train_set, eval_set, ):
		# train function


	def save_model(self):
		# save model


	def load_model(self):
		# load model

	
	def save_layers_wav(self):
		# save layers wave

	def sound_layer(self, layer_num):
		# convert layer to wav
		# play wav file

	def test_graph(self, test_set_provider, results=['confusion'])
		# test graph with unseen dataset

	
		

	
