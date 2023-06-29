#include "ann.h"

int main(int argc, char** argv)
{
	struct ann ann;
	double input[3] = {2.0, 1.0, 10.0};
	double output = 0.0;
	double dataset_in[30] = {0};
	double dataset_out[10] = {0};
	ann = ann_create(3, 2, 1);	// creating ANN with 3-2-1 structure
	
	// basic setup before training or running, if you do not setup the ANN, then it will use default setting
	ann->lrate = 0.05;	// setup learning rate
	ann->target = 6;	// setup target error
	ann->max_epoch = 10000;	// setup maximum epoch
	ann_hidden_func(&ann, ANN_RELU);	// setup activation function for hidden layer, check header file for more detail
	ann_output_func(&ann, ANN_SIGMOID);	// setup activation function for output layer, check for header file for more detail

	// training the ANN
	ann_teach(&ann, dataset_in, dataset_out, 10);	// the dataset have 10 records consists of 3 input 1 output for each record. so you need 30 input buffer and 10 output buffer

	// How to run the ANN
	// first you need add input data
	ann_add_input(&ann, input);
	// feed forward
	ann_forward(&ann);
	// read the output
	ann_get_output(&ann, &output);	
	return 0;
}