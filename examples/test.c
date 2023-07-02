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
		printf("%f\n", __coin_toss(0.02));
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
	//ann_export(&ann, "nino.ann");

	printf("\n\n\nTEST: ann_export()\n-------------------------\n");
	ann_export(&ann, "test.ann");
	printf("check your file system, it should be appeared a new file named test.ann\n");
	ann_destroy(&ann);
	
	struct ann* ann2 = ann_import("test.ann");
	printf("\n\n\nTEST: ann_import()\n-------------------------\n");
	printf("========== File contents: =========\n");
	FILE* f = fopen("test.ann", "r");
	char s[50];
	while( fgets(s, 50, f))
	{
		printf("%s", s);
	}
	fclose(f);
	
	printf("\n\n========Actual Object==========\n");
	__ann_print(ann2);
	ann_destroy(ann2);

	return 0;
}