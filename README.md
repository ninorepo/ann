# ann

As the name suggests it is library for creating artificial neural network in c. Structured as a simple to use library for generating single hidden layered network.

## Installation

Compile the ann library using this command:

```bash
gcc -c ann.c -lm
```

After that include the ann.h and link the ann.o into your project during linking process. Compiling command example:

```bash
gcc -o your_project your_project.c ann.o -lm
```


## Usage

### Creating and Destroying Object

```c
	struct ann ann = ann_create( 3, 2, 1 );	// creating ann with 3 input nodes, 2 hidden nodes,and 1 output node
	...
	...
	...
	ann_destroy(&ann);
```

### Basic Setup
```c
	ann.lrate = 0.05;	// learning rate
	ann.target = 0.3;	// target error you want to achieve
	ann.min_norm = 0;
	ann.max_norm = 100; // range number used for normalizing the input and output
	ann.hidden_func(&ann, ANN_RELU);	// set hidden activation function to RELU
	ann.output_func(&ann, ANN_SOFTMAX); // set output activation function to softmax
```

### Training
```c
	double dataset_in[30];	// the ann have 3 inputs, data set is consists of 10 data records
	double dataset_out[10]; // the ann has 1 output with 10 data records
	ann.teach( &ann, dataset_in, dataset_out, 10); // run the training process untill target error has been reached
```

### Accessing input and output nodes
```c
	double buffer_in[3] = {10.0,50.0,29.6};
	double buffer_out = 0.0;
	ann_add_input(&ann, buffer_in);	// adding 3 inputs to the ann,the input automatically normalized by this function
	ann_get_output(&ann, &buffer_out);	//store the result to a buffer, there is 1 output node in this example, the output was automatically denormalized by the function
```

### Saving the training result, export - import
```c
	ann_export(&ann, "path/file.ann");	// save ann data to a text file
	destroy(&ann);
	
	struct ann* new_ann = ann_import("path/file.ann"); // create new ann based on data from the text file saved before
```