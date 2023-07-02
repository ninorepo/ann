#define ANN_VERBOSE

#include "ann.h"
#include "stdlib.h"

void ann_print(struct ann *ann)
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
	printf("\n\n\n");
}

int main(int argc, char *argv[])
{
	double dataset_in[40] = {0.0};
	double dataset_out[20] = {0.0};
	struct ann ann = ann_create(2, 10, 1);
	double net_in[2] = {0.0};
	double net_out = 0.0;
	double calc = 0.0;
	double normalized = 0.0;
	ann.max_epoch = 50000000;
	ann.lrate = 0.005;
	ann.target = 90;
	ann.max_norm = 1000;
	ann.debug = 1;
	ann_hidden_func(&ann, ANN_LINEAR);
	ann_output_func(&ann, ANN_SIGMOID);

	//srand(50);
	int i;
	for (i = 0; i < 20; i++)
	{
		dataset_in[i * 2] = rand() % 10;
		dataset_in[i * 2 + 1] = rand() % 10;
		dataset_out[i] = dataset_in[i * 2] * dataset_in[i * 2 + 1];
	}

	// training
	ann_teach(&ann, dataset_in, dataset_out, 20);

	//print dataset
	printf("\n\n\nDataset\n=============\n");
	for (i = 0; i < 50; i++)
	{
		printf("%3.2f\tX\t%3.2f\t=\t%3.2f\n", dataset_in[i * 2], dataset_in[i * 2 + 1], dataset_out[i]);
	}

	//network run
	printf("\n\n\nTraining Result\n==========\n");
	printf("value1 X value2\t\texpected\tmodeled\t\tdiff\n\n");
	for (i = 0; i < 40; i++)
	{
		net_in[0] = i;
		net_in[1] = 2;
		ann_add_input(&ann, net_in);
		ann_forward(&ann);
		ann_get_output(&ann, &net_out);
		calc = i * 2;
		normalized = (calc - ann.min_norm) / (ann.max_norm - ann.min_norm);
		printf("%4.2f X %4.2f\t\t%f\t%f\t%f\n", (double)i, 2.0, calc, net_out, calc - net_out);
		//printf("%4.2f + %4.2f\t\t%f\t%f\t%f\n", (double)i, 2.0, normalized, ann.output[0], calc - net_out);
	}
}
