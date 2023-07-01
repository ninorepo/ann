#include "ann.h"

void __print_array(const char *label, double *arr, int size)
{
	int i;
	printf("%s: \n{", label);
	for (i = 0; i < size; i++)
	{
		printf("%2.2f  ", arr[i]);
	}
	printf("}\n");
}

void __ann_print(struct ann *ann)
{
	int i;
	printf("weights\t");
	for (i = 0; i < ann->weight_size; i++)
	{
		printf("%f  ", ann->weight[i]);
	}
	printf("\n");
	printf("inputs\t");
	for (i = 0; i < ann->input_size; i++)
	{
		printf("%f  ", ann->input[i]);
	}
	printf("\n");

	printf("hidden\t");
	for (i = 0; i < ann->hidden_size; i++)
	{
		printf("%f  ", ann->hidden[i]);
	}
	printf("\n");

	printf("output\t");
	for (i = 0; i < ann->output_size; i++)
	{
		printf("%f  ", ann->output[i]);
	}
	printf("\n");
	printf("input-size=%d\n", ann->input_size);
	printf("hidden-size=%d\n", ann->hidden_size);
	printf("output-size=%d\n", ann->output_size);
	printf("weight-size=%d\n", ann->weight_size);
	printf("learning-rate=%lf\n", ann->lrate);
	printf("error=%lf\n", ann->error);
	printf("max-epoch=%d\n", ann->max_epoch);
	printf("target=%lf\n", ann->target);
	printf("min-norm=%d\n", ann->min_norm);
	printf("max-norm=%d\n", ann->max_norm);
	printf("hidden-func-type=%d\n", ann->hidden_func_type);
	printf("output-func-type=%d\n", ann->output_func_type);
}

int __ann_tanh(double *buffer, int size)
{
	int i;
	for (i = 0; i < size; i++)
	{
		buffer[i] = tanh(buffer[i]);
	}
	return 0;
}

int __ann_linear(double *buffer, int size)
{
	return 0;
}

int __ann_softmax(double *buffer, int size)
{
	double sum = 0.0;
	int i;
	for (i = 0; i < size; i++)
	{
		sum += exp(buffer[i]);
	}

	for (i = 0; i < size; i++)
	{
		buffer[i] = exp(buffer[i]) / sum;
	}

	return 0;
}

int __ann_sigmoid(double *buffer, int size)
{
	int i;
	for (i = 0; i < size; i++)
	{
		buffer[i] = 1.0 / (1.0 + exp(-1 * buffer[i]));
	}
	return 0;
}

int __ann_relu(double *buffer, int size)
{
	int i;
	for (i = 0; i < size; i++)
	{
		buffer[i] = (buffer[i] > 0) ? buffer[i] : 0;
	}
	return 0;
}

int ann_output_func(struct ann *ann, enum ann_func func)
{
	if (!ann)
		return ANN_INVALID_OBJ;

	switch (func)
	{
	case ANN_TANH:
		ann->output_func = __ann_tanh;
		break;
	case ANN_SIGMOID:
		ann->output_func = __ann_sigmoid;
		break;
	case ANN_LINEAR:
		ann->output_func = __ann_linear;
		break;
	case ANN_RELU:
		ann->output_func = __ann_relu;
		break;
	case ANN_SOFTMAX:
		ann->output_func = __ann_softmax;
		break;
	default:
		return ANN_INVALID_FUNC;
	}
	ann->output_func_type = func;
	return 0;
}

int ann_hidden_func(struct ann *ann, enum ann_func func)
{
	if (!ann)
		return ANN_INVALID_OBJ;

	switch (func)
	{
	case ANN_TANH:
		ann->hidden_func = __ann_tanh;
		break;
	case ANN_SIGMOID:
		ann->hidden_func = __ann_sigmoid;
		break;
	case ANN_LINEAR:
		ann->hidden_func = __ann_linear;
		break;
	case ANN_RELU:
		ann->hidden_func = __ann_relu;
		break;
	case ANN_SOFTMAX:
		ann->hidden_func = __ann_softmax;
		break;
	default:
		return ANN_INVALID_FUNC;
	}
	ann->hidden_func_type = func;
	return 0;
}

// generate random values between -lrate to +lrate
double __coin_toss(double number)
{
	int r = 0;
	double toss = 0.0;
	r = (rand() + 1) % 2;
	toss = (r > 0) ? 1.0 : -1.0;
	return number * toss;
}
// generate random values using __coin_toss,store the result to roulette array
int __ann_spin_roulette(struct ann *ann)
{
	int i;
	if (!ann)
		return ANN_INVALID_OBJ;
	for (i = 0; i < ann->weight_size; i++)
	{
		ann->roulette[i] = __coin_toss(ann->lrate);
	}
	return 0;
}

int __ann_default_setup(struct ann *ann)
{
	if (!ann)
		return ANN_INVALID_OBJ;
	ann->error = DBL_MAX;
	ann->lrate = 0.2;
	ann->target = 0.03;
	ann->epoch = 0;
	ann->max_epoch = 1000;
	ann->hidden_func = __ann_relu;
	ann->output_func = __ann_sigmoid;
	ann->hidden_func_type = ANN_RELU;
	ann->output_func_type = ANN_SIGMOID;
	ann->seed = time(NULL);
	ann->min_norm = 0;
	ann->max_norm = 1000;
	ann->debug = 0;
	//	ann->timeout = 10000;
	__ann_spin_roulette(ann);
	return 0;
}

// doing cross product between input layer to the adjescent weights
int __ann_input_x_weight(struct ann *ann)
{
	int i;
	int j;
	int w = 0;
	if (!ann)
		return ANN_INVALID_OBJ;
	for (i = 0; i < ann->hidden_size - 1; i++)
	{
		ann->hidden[i] = 0.0;
		for (j = 0; j < ann->input_size; j++)
		{
			ann->hidden[i] += ann->input[j] * ann->weight[w];
			++w;
		}
	}
	return 0;
}

// doing cross product between hidden layer to the adjescent weights
int __ann_hidden_x_weight(struct ann *ann)
{
	int i;
	int j;
	int w = ann->input_size * (ann->hidden_size - 1);
	if (!ann)
		return ANN_INVALID_OBJ;
	for (i = 0; i < ann->output_size; i++)
	{
		ann->output[i] = 0.0;
		for (j = 0; j < ann->hidden_size; j++)
		{
			ann->output[i] += ann->hidden[j] * ann->weight[w];
			++w;
		}
	}
	return 0;
}

int __ann_init_weight(struct ann *ann, double *array, int size)
{
	int randx = 0;
	int i;
	if (!ann)
		return ANN_INVALID_OBJ;
	if (!array)
		return ANN_INVALID_OBJ;
	for (i = 0; i < size; i++)
	{
		randx = rand();
		array[i] = (randx % 300) / 1000.0; // giving rand nunber between 0 - 0.3
	}
	return 0;
}

// basically this function add weights to the roulette array
int __ann_update(struct ann *ann)
{
	int i;
	if (!ann)
		return ANN_INVALID_OBJ;
	for (i = 0; i < ann->weight_size; i++)
	{
		ann->weight[i] += ann->roulette[i];
	}
	return 0;
}

// revert the value of weights to the previous values,cancel out weight change
int __ann_reset(struct ann *ann)
{
	int i;
	if (!ann)
		return ANN_INVALID_OBJ;
	for (i = 0; i < ann->weight_size; i++)
	{
		ann->weight[i] -= ann->roulette[i];
	}
	return 0;
}

// loss function calculation, return the total error of an output array/nodes
double __ann_eval_loss(struct ann *ann, double *outputset)
{
	double denormalized = 0.0;
	double loss = 0.0;

	if (!ann)
		return ANN_INVALID_OBJ;
	if (!outputset)
		return ANN_INVALID_OBJ;
	int i;
	for (i = 0; i < ann->output_size; i++)
	{
		// convert the normalized output into the original value before compared to dataset/expected output
		denormalized = ann->output[i] * (ann->max_norm - ann->min_norm) + ann->min_norm;
		loss += abs(outputset[i] - denormalized);
	}
	//printf("loss: %f\n", loss);
	return loss;
}

// add input to the ANN in normalized form
int ann_add_input(struct ann *ann, double *input)
{
	int i;
	if (!ann)
		return ANN_INVALID_OBJ;
	if (!input)
		return ANN_INVALID_OBJ;
	for (i = 0; i < ann->input_size - 1; i++)
	{
		// normalize the input
		ann->input[i] = (input[i] - ann->min_norm) / (ann->max_norm - ann->min_norm);
	}
	return 0;
}

// get output and denormalize the output
int ann_get_output(struct ann *ann, double *buffer)
{
	int i;
	if (!ann)
		return ANN_INVALID_OBJ;
	if (!buffer)
		return ANN_INVALID_OBJ;
	for (i = 0; i < ann->output_size; i++)
	{
		buffer[i] = ann->output[i] * (ann->max_norm - ann->min_norm) + ann->min_norm;
	}
	return 0;
}

struct ann ann_create(int in, int hid, int out)
{
	struct ann ann;

	ann.seed = time(NULL);
	srand(ann.seed);

	ann.input_size = in + 1;
	ann.hidden_size = hid + 1;
	ann.output_size = out;
	ann.weight_size = ann.input_size * (ann.hidden_size - 1) + ann.hidden_size * ann.output_size;

	ann.input = (double *)malloc(sizeof(double) * ann.input_size);
	ann.hidden = (double *)malloc(sizeof(double) * ann.hidden_size);
	ann.output = (double *)malloc(sizeof(double) * ann.output_size);
	ann.weight = (double *)malloc(sizeof(double) * ann.weight_size);
	ann.roulette = (double *)calloc(sizeof(double), ann.weight_size);

	ann.input[ann.input_size - 1] = 1.0;
	ann.hidden[ann.hidden_size - 1] = 1.0;
	__ann_init_weight(&ann, ann.weight, ann.weight_size);
	__ann_default_setup(&ann);

	return ann;
}

int ann_destroy(struct ann *ann)
{
	if (!ann)
		return ANN_INVALID_OBJ;
	free(ann->input);
	free(ann->hidden);
	free(ann->output);
	free(ann->weight);
	free(ann->roulette);
	return 0;
}

// doing feedforward
int ann_forward(struct ann *ann)
{
	if (!ann)
		return ANN_INVALID_OBJ;
	__ann_input_x_weight(ann);
	__ann_hidden_x_weight(ann);
	return 0;
}

int ann_teach(struct ann *ann, double *inputset, double *outputset, int record_num)
{
	int epoch = 0;
	int loops = 0;
	double e = 0.0; // average error of one epoch
	int i = 0;
	int j = 0;
	//	int reset_timeout = ann->timeout;

	if (!ann)
		return ANN_INVALID_OBJ;
	if (!inputset)
		return ANN_INVALID_OBJ;
	if (!outputset)
		return ANN_INVALID_OBJ;
	if (record_num <= 0)
		return ANN_INVALID_VAL;

	while (1)
	{
		// break if target error has been achieved, or max epoch reached
		if (ann->error <= ann->target)
			break;
		if(ann->epoch >= ann->max_epoch)
			break;

		// if one epoch has been reached
		if (loops >= record_num)
		{
			++ann->epoch;
			if(ann->debug)			
				printf("epoch: %d\t", ann->epoch);
			//e = e/(record_num*ann->output_size);
			i = 0;
			j = 0;
			loops = 0;
			// code for update and reset weights here ...
			if ((e < ann->error)) // if new_error smaller than old_error
			{
				// update
				__ann_update(ann);
				ann->error = e;
				if(ann->debug)
					printf("e: %f\n", ann->error);
			}
			else
			{
				__ann_reset(ann);		  // reset or revert weights back to previous values
				__ann_spin_roulette(ann); // spin roulette, roulette is array of values used for increasing or decreasing weights
				__ann_update(ann);		  // update weights
				if(ann->debug)				
					printf("e: %f\n", ann->error);
			}
			e = 0.0;
		}

		// feed forward
		ann_add_input(ann, inputset + i);
		ann_forward(ann);

		//error calculation
		e += __ann_eval_loss(ann, outputset + j);

		// striding to the next record
		i += ann->input_size - 1;
		j += ann->output_size;

		//loop counter
		++loops;
	}
	return 0;
}

int ann_export(struct ann *ann, char *filepath)
{
	FILE *f = fopen(filepath, "w");

	if (!f)
		return ANN_FILE_FAILURE;
	fprintf(f, "input-size=%d\r\n", ann->input_size);
	fprintf(f, "hidden-size=%d\r\n", ann->hidden_size);
	fprintf(f, "output-size=%d\r\n", ann->output_size);
	fprintf(f, "weight-size=%d\r\n", ann->weight_size);
	fprintf(f, "learning-rate=%f\r\n", ann->lrate);
	fprintf(f, "error=%f\r\n", ann->error);
	fprintf(f, "max-epoch=%d\r\n", ann->max_epoch);
	fprintf(f, "target=%f\r\n", ann->target);
	fprintf(f, "min-norm=%d\r\n", ann->min_norm);
	fprintf(f, "max-norm=%d\r\n", ann->max_norm);
	fprintf(f, "hidden-func-type=%d\r\n", (int)ann->hidden_func_type);
	fprintf(f, "output-func-type=%d\r\n", (int)ann->output_func_type);
	fprintf(f, "#weight-start\r\n");

	int i;
	for (i = 0; i < ann->weight_size; i++)
	{
		fprintf(f, "%f\r\n", ann->weight[i]);
	}
	fprintf(f, "#weight-end\r\n");
	fclose(f);
	return 0;
}

struct ann *ann_import(const char *filepath)
{
	char str[35] = {0};
	int input_size = 0;
	int hidden_size = 0;
	int output_size = 0;
	int weight_size = 0;
	double lrate = 0.0;
	double error = 0.0;
	int max_epoch = 0;
	double target = 0.0;
	int min_norm = 0;
	int max_norm = 0;
	enum ann_func hidden_func_type = ANN_RELU;
	enum ann_func output_func_type = ANN_RELU;
	int weight_cursor = 0;
	int weight_start = 0;
	double weight = 0.0;

	struct ann *ann = NULL;
	FILE *f = fopen(filepath, "r");

	if (!f)
		return NULL;

	while (fgets(str, 35, f) != NULL)
	{
		sscanf(str, "input-size=%d", &input_size);
		sscanf(str, "hidden-size=%d", &hidden_size);
		sscanf(str, "output-size=%d", &output_size);
		sscanf(str, "weight-size=%d", &weight_size);
		sscanf(str, "learning-rate=%lf", &lrate);
		sscanf(str, "error=%lf", &error);
		sscanf(str, "max-epoch=%d", &max_epoch);
		sscanf(str, "target=%lf", &target);
		sscanf(str, "min-norm=%d", &min_norm);
		sscanf(str, "max-norm=%d", &max_norm);
		sscanf(str, "hidden-func-type=%d", (int*)&hidden_func_type);
		sscanf(str, "output-func-type=%d", (int*)&output_func_type);

		// create neural network when all requirements have been retrieved: input size,hidden size,output size
		if (output_size && !ann)
		{
			ann = (struct ann *)malloc(sizeof(struct ann));
			*ann = ann_create(input_size - 1, hidden_size - 1, output_size);
		}
		// detect weight start tag to extract weights
		if (strcmp(str, "#weight-start\r\n") == 0)
		{
			weight_start = 1;
			fgets(str, 35, f); // skip #weight-start tag in *.ann file
		}
		if (strcmp(str, "#weight-end\r\n") == 0)
			weight_start = 0;

		if (weight_start)
		{
			sscanf(str, "%lf", &weight);
			ann->weight[weight_cursor] = weight;
			++weight_cursor; //weight_cursor is used as index of weight array in the ann object
		}
	}
	// transfer datas into ANN object
	ann->lrate = lrate;
	ann->error = error;
	ann->max_epoch = max_epoch;
	ann->target = target;
	ann->min_norm = min_norm;
	ann->max_norm = max_norm;
	ann->hidden_func_type = (enum ann_func)hidden_func_type;
	ann->output_func_type = (enum ann_func)output_func_type;
	
	fclose(f);
	return ann;
}