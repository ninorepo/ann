#ifndef ANN_H
#define ANN_H

#include "string.h" 	//memcpy, strcmp
#include "stdlib.h" 	// malloc
#include "time.h"	// time(NULL)
#include "math.h"	// tanh
#include "float.h"	//DBL_MAX
#include "stdio.h"	//printf

enum ann_func
{
	ANN_TANH,
	ANN_SIGMOID,
	ANN_RELU,
	ANN_SOFTMAX,
	ANN_LINEAR
};

enum ann_err
{
	ANN_OK,
	ANN_INVALID_OBJ,
	ANN_INVALID_VAL,
	ANN_FILE_FAILURE,
	ANN_INVALID_FUNC
};

struct ann
{
	double *input;
	double *hidden;
	double *output;
	double *weight;
	//double *bias;
	double *roulette;

	int input_size;
	int hidden_size;
	int output_size;
	int weight_size;
	//int bias_size;
	int roulette_size;

	double error;  // latest error value
	double lrate;  // learning rate
	int epoch;
	int max_epoch; // max epoch
	double target; // target error

	enum ann_func hidden_func_type;
	enum ann_func output_func_type;
	int (*hidden_func)(double *array, int size);
	int (*output_func)(double *array, int size);

	int seed;
	//	int timeout;

	int min_norm;
	int max_norm;

	int debug;
};

void __print_array(const char *label, double *arr, int size);
void __ann_print(struct ann *ann);;
int __ann_linear(double *buffer, int size);
int __ann_softmax(double *buffer, int size);
int __ann_sigmoid(double *buffer, int size);
int __ann_relu(double *buffer, int size);
int ann_output_func(struct ann *ann, enum ann_func func);
int ann_hidden_func(struct ann *ann, enum ann_func func);
double __coin_toss(double number);				// generate random values between -lrate to +lrate
int __ann_spin_roulette(struct ann *ann);			// generate random values using __coin_toss,store the result to roulette array
int __ann_default_setup(struct ann *ann);
int __ann_input_x_weight(struct ann *ann);			// doing cross product between input layer to the adjescent weights
int __ann_hidden_x_weight(struct ann *ann);			// doing cross product between hidden layer to the adjescent weights
int __ann_init_weight(struct ann *ann, double *array, int size);
int __ann_update(struct ann *ann);				// basically this function add weights to the roulette array
int __ann_reset(struct ann *ann);				// revert the value of weights to the previous values,cancel out weight change
double __ann_eval_loss(struct ann *ann, double *outputset);	// loss function calculation, return the total error of an output array/nodes
int ann_add_input(struct ann *ann, double *input);		// add input to the ANN in normalized form
int ann_get_output(struct ann *ann, double *buffer);		// get output and denormalize the output
struct ann ann_create(int in, int hid, int out);
int ann_destroy(struct ann *ann);
int ann_forward(struct ann *ann);				// doing feedforward
int ann_teach(struct ann *ann, double *inputset, double *outputset, int record_num);
int ann_export(struct ann *ann, char *filepath);		// save ann object datas into text file
struct ann *ann_import(const char *filepath);			// create ann object based on previously saved data

#endif
