#include "ann.h"

int main(int argc, char *argv[])
{
	int i;
	struct ann ann = ann_create(3, 2, 1);
	double input[3] = {10,20,30};

	printf("TEST: structure\n--------------\n");
	printf("input: %2.2f  %2.2f  %2.2f  %2.2f\n", ann.input[0], ann.input[1], ann.input[2], ann.input[3]);
	printf("weight: ");
	for (i = 0; i < 8; i++)
	{
		printf("%2.2f  ", ann.weight[i]);
	}
	printf("\n");
	printf("hidden: %2.2f  %2.2f %2.2f\n", ann.hidden[0], ann.hidden[1], ann.hidden[2]);
	printf("weight: ");
	for (i = 0; i < 3; i++)
	{
		printf("%2.2f  ", ann.weight[8 + i]);
	}
	printf("\n");
	printf("output: %2.2f\n", ann.output[0]);

	printf("\n\n\nTEST: ann_forward()\n---------------\n");
	ann_forward(&ann);
	printf("input: %2.2f  %2.2f  %2.2f  %2.2f\n", ann.input[0], ann.input[1], ann.input[2], ann.input[3]);
	printf("weight: ");
	for (i = 0; i < 8; i++)
	{
		printf("%2.2f  ", ann.weight[i]);
	}
	printf("\n");
	printf("hidden: %2.2f  %2.2f %2.2f\n", ann.hidden[0], ann.hidden[1], ann.hidden[2]);
	printf("weight: ");
	for (i = 0; i < 3; i++)
	{
		printf("%2.2f  ", ann.weight[8 + i]);
	}
	printf("\n");
	printf("output: %2.2f\n", ann.output[0]);
	
	printf("\n\n\nTEST: __coin_toss()\n--------------\n");
	for(i=0 ; i<10 ; i++)
	{
		printf("%f\n", __coin_toss(0.4));
	}
	
	printf("\n\n\nTEST: ann_add_input(), normalized input\n--------------\n");
	ann_add_input(&ann, input);
	ann_forward(&ann);
	printf("input %f %f %f\n", input[0], input[1], input[2]);
	printf("norm input: %2.2f  %2.2f  %2.2f  %2.2f\n", ann.input[0], ann.input[1], ann.input[2], ann.input[3]);
	printf("weight: ");
	for (i = 0; i < 8; i++)
	{
		printf("%2.2f  ", ann.weight[i]);
	}
	printf("\n");
	printf("hidden: %2.2f  %2.2f %2.2f\n", ann.hidden[0], ann.hidden[1], ann.hidden[2]);
	printf("weight: ");
	for (i = 0; i < 3; i++)
	{
		printf("%2.2f  ", ann.weight[8 + i]);
	}
	printf("\n");
	printf("output: %2.2f\n", ann.output[0]);

	return 0;
}