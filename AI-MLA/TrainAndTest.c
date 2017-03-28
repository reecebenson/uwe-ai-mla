//
//  TrainAndTest.c
//  MLCoursework
//
//  This is a fairly inefficient implentation that does not use any dynamic memory allocation
//  because that would not be safe on the DEWIS marking system
//
//  Created by Jim Smith on 06/02/2017.
//  Copyright 2017 Jim Smith. All rights reserved.
//

// > Includes
#include "TrainAndTest.h"
#include <math.h>

// > Default Variables
static double	myModel[NUM_TRAINING_SAMPLES][NUM_FEATURES];
static char		myModelLabels[NUM_TRAINING_SAMPLES];
static int		trainingSetSize = 0;
static int		isDebugging = 1;
static int		isShowingPredictions = 0;

// > Category Struct, used to keep track of how many ACTUAL categories there are
struct CategoryLabels
{
	int		count;
	int		uniary_index;
	char	actual_label;
} static category_labels[NUM_TRAINING_SAMPLES];

// > Neural Network Variables
static double	myModelNN[NUM_TRAINING_SAMPLES][NUM_FEATURES * 2];
static int		curindex = 0;
static int		numInput = NUM_FEATURES;
static int		numHidden = 3;
static int		numOutput;
static double	inputs[NUM_FEATURES];
static double	res[NUM_FEATURES];
static double	w_res[NUM_SAMPLES];

// > Rates
static int		curEpoch = 0;
static int		maxEpoch = 23500;// 5380;
static double	alpha = 0.24;
static double	momentum = 0.005;

// > Input -> Hidden
static double	ihWeights[NUM_FEATURES][NUM_FEATURES];
static double	hBiases[NUM_FEATURES];
static double	hOutputs[NUM_FEATURES];

// > Hidden -> Output
static double	hoWeights[NUM_FEATURES][NUM_FEATURES];
static double	oBiases[NUM_FEATURES];
static double	oOutputs[NUM_FEATURES];

// > Neural Network Functions
// -> Main
static void		nn_init();
static double*	nn_train();
static void		nn_generateweights();
static double*	nn_getweights();
static double*	nn_compute(double* xVals);
static double	nn_error();
static void		nn_shuffle(int* seq);
static int		nn_maxindex(double* vector);
static int		nn_predict(double* sample);
// -> Debugging
static void		nn_showinfo();
static void		nn_printnnmatrix();
static void		nn_printprediction(int uniary, char lbl);
static void		nn_displayweights();
// -> Maths
static double	nn_hypertan(double x);
static double*	nn_softmax(double* sums);
int				rand(void);
double			fRand(double fMin, double fMax);

// > Train
int train(double **trainingSamples, char *trainingLabels, int numSamples, int numFeatures)
{
	// > Variables
	int returnVal = 1, s, f, c, m;

	// > Clean the model because C leaves whatever is in the memory
	for (s = 0; s < NUM_TRAINING_SAMPLES; s++)
	{
		for (f = 0; f < NUM_FEATURES; f++)
			myModel[s][f] = 0.0;
	}

	// > Sanity check
	if (numFeatures > NUM_FEATURES || numSamples > NUM_TRAINING_SAMPLES)
	{
		fprintf(stdout, "[TRAIN] -> Error: called train with data set larger than spaced allocated to store it.\n");
		returnVal = 0;
	}

	// > Check our sanity was a success
	if (returnVal == 1)
	{
		// > Default implementation of sorting data into myModel and myModelLabels
		trainingSetSize = numSamples;
		for (s = 0; s < numSamples; s++)
		{
			myModelLabels[s] = trainingLabels[s];
			for (f = 0; f < numFeatures; f++)
			{
				myModel[s][f] = trainingSamples[s][f];
			}
		}

		// > Work out how many categories there are
		for (s = 0; s < numSamples; s++)
		{
			char lbl = myModelLabels[s];
			int newLabelFound = 1;
			for (c = 0; c < NUM_TRAINING_SAMPLES; c++)
			{
				// > A new label has been found
				if (category_labels[c].actual_label == lbl)
				{
					newLabelFound = 0;
					category_labels[c].count++;
				}
			}

			if (newLabelFound)
			{
				category_labels[numOutput].count = 1;
				category_labels[numOutput].actual_label = lbl;
				category_labels[numOutput].uniary_index = numOutput;
				if (isDebugging) fprintf(stdout, "Label found: %c -> Uniary Index: %d\n", lbl, numOutput);
				numOutput++;
			}
		}

		// > Change our model array to hold data in this format: F=(5.1, 3.5, 1.4, 0.2) V=(1, 0, 0)
		// -> Fill in myModelNN with zero data
		for (s = 0; s < numSamples; s++)
		{
			for (f = 0; f < numFeatures; f++)
			{
				myModelNN[s][f] = 0.0;
			}
		}

		// > Go through our model
		for (s = 0; s < numSamples; s++)
		{
			// > Add our sample to the NN model
			for (f = 0; f < numFeatures; f++)
			{
				myModelNN[s][f] = trainingSamples[s][f];
			}

			// > Figure out what category this sample goes into
			char lbl = myModelLabels[s];
			for (c = 0; c < NUM_TRAINING_SAMPLES; c++)
			{
				// > A new label has been found
				if (category_labels[c].actual_label == lbl && category_labels[c].actual_label != '\0')
				{
					// > Add our category uniary data to the NN model
					for (f = numFeatures, m = 0; f < (numFeatures + numOutput); f++, m++)
					{
						if (m == category_labels[c].uniary_index)
							myModelNN[s][f] = 1;
						else
							myModelNN[s][f] = 0;
					}

					// > Debug
					if (isDebugging) fprintf(stdout, "Sample label: %c -> Found uniary index: %d\n", category_labels[c].actual_label, category_labels[c].uniary_index);
					break;
				}
			}
		}

		// > Debug
		nn_printnnmatrix();

		// > Initialise our neural network
		nn_init();
	}

	// > Return our state
	return returnVal;
}

// > Predict
char  predictLabel(double *sample, int numFeatures)
{
	// > Variables
	int s;
	char prediction = 'a';

	// > Find our prediction
	if (isDebugging && isShowingPredictions) fprintf(stdout, "%d", curindex);
	int pred = nn_predict(sample);

	// > Find our label from our prediction (uniary index)
    for (s = 0; s < NUM_FEATURES*2; s++)
	{
		int cur_uniary = category_labels[s].uniary_index;
		if (pred == cur_uniary)
		{
			prediction = category_labels[s].actual_label;
			nn_printprediction(pred, prediction);
			break;
		}
	}

	// > Keep track of where we are
	curindex++;
	if (curindex == NUM_TRAINING_SAMPLES) curindex = 0;

	// > Spacer
	if (isDebugging && isShowingPredictions) fprintf(stdout, "\n");

	// > Return our prediction
	return prediction;
}

// > Functions
// -> Calculate random value
double fRand(double fMin, double fMax)
{
	double f = (double)rand() / 2147483647; // INT_MAX
	return fMin + f * (fMax - fMin);
}

// --------------------------------------------- //
// Neural Network Implementation by Reece Benson //
// --------------------------------------------- //
static void nn_init()
{
	// > Generate our weights
	nn_generateweights();
	nn_displayweights();

	// > Show our debug information
	nn_showinfo();
	nn_train();
	nn_displayweights();
}
static double* nn_train()
{
	// > Train Data
	double  trainData[NUM_TRAINING_SAMPLES][NUM_FEATURES * 2];

	// > This training method using Back Propagation
	double  hoGrads[NUM_FEATURES * 2][NUM_FEATURES * 2];
	double  ihGrads[NUM_FEATURES * 2][NUM_FEATURES * 2];
	double  obGrads[NUM_FEATURES * 2];
	double  hbGrads[NUM_FEATURES * 2];
	double oSignals[NUM_FEATURES * 2];
	double hSignals[NUM_FEATURES * 2];

	// > Back Prop Momentum
	double ihPrevWeightsDelta[NUM_FEATURES * 2][NUM_FEATURES * 2];
	double hoPrevWeightsDelta[NUM_FEATURES * 2][NUM_FEATURES * 2];
	double hPrevBiasesDelta[NUM_FEATURES * 2];
	double oPrevBiasesDelta[NUM_FEATURES * 2];

	// > Variables
	double xValues[NUM_FEATURES];	// < Input Values
	double tValues[NUM_FEATURES * 2]; // < Target Values
	double derivative = 0.0, errorSignal = 0.0;
	int x, i, f, j, k;
	curEpoch = 0;

	// > Set "trainData" as myModelNN
	for (i = 0; i < NUM_TRAINING_SAMPLES; i++)
	{
		for (j = 0; j < NUM_FEATURES + numOutput; j++)
		{
			trainData[i][j] = myModelNN[i][j];
		}
	}

	// > Give our new arrays some data
	for (i = 0; i < NUM_FEATURES * 2; i++)
	{
		for (j = 0; j < NUM_FEATURES * 2; j++)
		{
			hoGrads[i][j] = 0.0;
			ihGrads[i][j] = 0.0;
			ihPrevWeightsDelta[i][j] = 0.0;
			hoPrevWeightsDelta[i][j] = 0.0;
		}

		hPrevBiasesDelta[i] = 0.0;
		oPrevBiasesDelta[i] = 0.0;
		obGrads[i] = 0.0;
		hbGrads[i] = 0.0;
		oSignals[i] = 0.0;
		hSignals[i] = 0.0;
	}

	// > Shuffler
	int sequence[NUM_TRAINING_SAMPLES];
	for (i = 0; i < NUM_TRAINING_SAMPLES; i++)
		sequence[i] = i;

	// > Train on our data
	int errInterval = maxEpoch / 10; // interval to check error
	while (curEpoch < maxEpoch)
	{
		// > Increment our epoch
		curEpoch++;

		// > Debug our epoch
		if (isDebugging && ((curEpoch % errInterval) == 0 && curEpoch < maxEpoch))
		{
			double trainErr = nn_error();
			fprintf(stdout, "XX |   Epoch = %d, Error = %f\n", curEpoch, trainErr);
		}

		// > Shuffle our sequence (to check our training data in a random order)
		nn_shuffle(sequence);

		// > Go through our training samples
		int id;
		for (x = 0; x < NUM_TRAINING_SAMPLES; x++)
		{
			// > Set our ID
			id = sequence[x];

			// > Move our features to xValues
			for (f = 0; f < NUM_FEATURES; f++)
				xValues[f] = trainData[id][f];

			// > Move our targets to tValues
			for (f = NUM_FEATURES, j = 0; f < (NUM_FEATURES + numOutput); f++, j++)
				tValues[j] = trainData[id][f];

			// > Compute our outputs
			nn_compute(xValues);
			// > Perform our feed forward & back propagation
			// [NOTE]: i = inputs, j = hidden, k = outputs

			// > [1] > Compute output node signals
			for (k = 0; k < numOutput; k++)
			{
				errorSignal = tValues[k] - oOutputs[k];
				derivative = (1 - oOutputs[k]) * oOutputs[k];
				oSignals[k] = errorSignal * derivative;
			}

			// > [2] > Hidden to Output weight
			for (j = 0; j < numHidden; j++)
			{
				for (k = 0; k < numOutput; k++)
				{
					hoGrads[j][k] = oSignals[k] * hOutputs[j];
				}
			}

			// > [2b] > Compute output bias gradients
			for (k = 0; k < numOutput; k++)
			{
				obGrads[k] = oSignals[k] * 1.0;
			}

			// > [3] > Compute hidden node signals
			for (j = 0; j < numHidden; j++)
			{
				derivative = (1 + hOutputs[j]) * (1 - hOutputs[j]);
				double sum = 0.0;
				for (k = 0; k < numOutput; k++)
				{
					sum += oSignals[k] * hoWeights[j][k];
				}
				hSignals[j] = derivative * sum;
			}

			// > [4] > Compute Input to Hidden weight gradients
			for (i = 0; i < numInput; i++)
			{
				for (j = 0; j < numHidden; j++)
				{
					ihGrads[i][j] = hSignals[j] * inputs[i];
				}
			}

			// > [4b] > Compute hidden node bias gradients
			for (j = 0; j < numHidden; j++)
			{
				hbGrads[j] = hSignals[j] * 1.0;
			}

			///////////////////////////////////
			// > Update our weights & biases //
			///////////////////////////////////

			// > Update Input to Hidden Weights
			for (i = 0; i < numInput; i++)
			{
				for (j = 0; j < numHidden; j++)
				{
					double delta = ihGrads[i][j] * alpha;
					ihWeights[i][j] += delta;
					ihWeights[i][j] += ihPrevWeightsDelta[i][j] * momentum;
					ihPrevWeightsDelta[i][j] = delta;
				}
			}

			// > Update Hidden Biases
			for (j = 0; j < numHidden; j++)
			{
				double delta = hbGrads[j] * alpha;
				hBiases[j] += delta;
				hBiases[j] += hPrevBiasesDelta[j] * momentum;
				hPrevBiasesDelta[j] = delta;
			}

			// > Update Hidden to Output Weights
			for (j = 0; j < numHidden; j++)
			{
				for (k = 0; k < numOutput; k++)
				{
					double delta = hoGrads[j][k] * alpha;
					hoWeights[j][k] += delta;
					hoWeights[j][k] += hoPrevWeightsDelta[j][k] * momentum;
					hoPrevWeightsDelta[j][k] = delta;
				}
			}

			// > Update Output Biases
			for (int k = 0; k < numOutput; ++k)
			{
				double delta = obGrads[k] * alpha;
				oBiases[k] += delta;
				oBiases[k] += oPrevBiasesDelta[k] * momentum;
				oPrevBiasesDelta[k] = delta;
			}
		}
	}

	// > Debug our final epoch
	if (isDebugging && ((curEpoch % errInterval) == 0))
	{
		double trainErr = nn_error();
		fprintf(stdout, "XX |   Epoch = %d, Error = %f\n", curEpoch, trainErr);
	}

	double* bestWeights = nn_getweights();
	return bestWeights;
}

static void nn_generateweights()
{
	// > Variables
	int w, i, j, k = 0;
	int totalWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
	double weights[NUM_SAMPLES];
	for (w = 0; w < totalWeights; w++)
		weights[w] = fRand(-0.1, 0.1);

	// > Set our weights from our generated values
	// -> Set our input -> hidden weights
	for (i = 0; i < numInput; i++)
		for (j = 0; j < numHidden; j++)
			ihWeights[i][j] = weights[k++];

	// -> Hidden biases
	for (i = 0; i < numHidden; i++)
		hBiases[i] = weights[k++];

	// -> Set our hidden -> output weights
	for (i = 0; i < numHidden; i++)
		for (j = 0; j < numOutput; j++)
			hoWeights[i][j] = weights[k++];

	// -> Output biases
	for (i = 0; i < numOutput; i++)
		oBiases[i] = weights[k++];
}

static double* nn_getweights()
{
	// > Variables
	int i, j, k = 0;
	int totalWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;

	// > Clear our w_res array
	for (i = 0; i < totalWeights; i++)
	{
		w_res[i] = 0.0;
	}

	// > Push our weights into 'res' array
	// -> Input -> Hidden weights
	for (i = 0; i < numInput; i++)
	{
		for (j = 0; j < numHidden; j++)
		{
			w_res[k++] = ihWeights[i][j];
		}
	}

	// -> Hidden biases
	for (i = 0; i < numHidden; i++)
	{
		w_res[k++] = hBiases[i];
	}

	// -> Hidden -> Output weights
	for (i = 0; i < numHidden; i++)
	{
		for (int j = 0; j < numOutput; j++)
		{
			w_res[k++] = hoWeights[i][j];
		}
	}

	// -> Output biases
	for (i = 0; i < numOutput; i++)
	{
		w_res[k++] = oBiases[i];
	}

	// > Return our weights
	return w_res;
}

static double* nn_compute(double* xVals)
{
	// > Variables
	int i, j;
	double hSums[NUM_FEATURES];
	double oSums[NUM_FEATURES];

	// > Set our values
	for (i = 0; i < NUM_FEATURES; i++)
	{
		hSums[i] = 0.0;
		oSums[i] = 0.0;
		res[i] = 0.0;
	}

	// > Copy xValues to Inputs
	for (i = 0; i < NUM_FEATURES; i++)
	{
		inputs[i] = xVals[i];
	}

	// > Compute Input to Hidden sum
	for (j = 0; j < numHidden; j++)
	{
		for (i = 0; i < numInput; i++)
		{
			hSums[j] += inputs[i] * ihWeights[i][j];
		}
	}

	// > Add biases to hidden nodes (hSums)
	for (i = 0; i < numHidden; i++)
	{
		hSums[i] += hBiases[i];
	}

	// > Apply Sigmoid to hidden nodes
	for (i = 0; i < numHidden; i++)
	{
		hOutputs[i] = nn_hypertan(hSums[i]);
	}

	// > Compute Hidden to Output sum
	for (j = 0; j < numOutput; j++)
	{
		for (i = 0; i < numHidden; i++)
		{
			oSums[j] += hOutputs[i] * hoWeights[i][j];
		}
	}

	// > Add biases to output nodes (oSums)
	for (i = 0; i < numOutput; i++)
	{
		oSums[i] += oBiases[i];
	}

	// > Apply softmax
	double* softOut = nn_softmax(oSums);
	for (i = 0; i < numOutput; i++)
		oOutputs[i] = softOut[i];

	// > Generate our reuslt
	for (i = 0; i < numOutput; i++)
	{
		res[i] = oOutputs[i];
	}

	return res;
}

static double nn_error()
{
	// > Variables
	int i, f, j;
	double sumSquaredError = 0.0;
	double xValues[NUM_FEATURES];
	double tValues[NUM_FEATURES * 2];

	// > Train Data
	double  trainData[NUM_TRAINING_SAMPLES][NUM_FEATURES * 2];

	// > Set "trainData" as myModelNN
	for (i = 0; i < NUM_TRAINING_SAMPLES; i++)
	{
		for (j = 0; j < NUM_FEATURES + numOutput; j++)
		{
			trainData[i][j] = myModelNN[i][j];
		}
	}

	// > Go through our data set
	for (i = 0; i < NUM_TRAINING_SAMPLES; i++)
	{
		// > Move our features to xValues
		for (f = 0; f < numInput; f++)
			xValues[f] = trainData[i][f];

		// > Move our targets to tValues
		for (f = numInput, j = 0; f < (numInput + numOutput); f++, j++)
			tValues[j] = trainData[i][f];

		// > Compute our outputs using the current weights
		double* yValues = nn_compute(xValues);
		for (j = 0; j < numOutput; j++)
		{
			double err = tValues[j] - yValues[j];
			sumSquaredError += err * err;
		}
	}

	// > Return our error
	return sumSquaredError / NUM_TRAINING_SAMPLES;
}

static void nn_shuffle(int* seq)
{
	// > Variables
	int i;

	// > Shuffle our sequence
	for (i = 0; i < NUM_TRAINING_SAMPLES; i++)
	{
		int r = (int)floor(fRand(i, NUM_TRAINING_SAMPLES));

		int tmp = seq[r];
		seq[r] = seq[i];
		seq[i] = tmp;
	}
}

static int nn_maxindex(double* vector)
{
	// index of largest value
	int bigIndex = 0;
	double biggestVal = vector[0];
	for (int i = 0; i < NUM_FEATURES; ++i)
	{
		if (vector[i] > biggestVal)
		{
			biggestVal = vector[i];
			bigIndex = i;
		}
	}
	return bigIndex;
}

static int nn_predict(double* sample)
{
	// > Variables
	int f;
	double xValues[NUM_FEATURES];

	// > Generate our outputs (predict) from our sample
	for (f = 0; f < NUM_FEATURES; f++)
		xValues[f] = sample[f];

	// > Find our prediction values
	double* yValues = nn_compute(xValues);

	// > Debug
	if (isDebugging)
	{
		// > Print our sample
		fprintf(stdout, " |   Sample Input: [");
		for (f = 0; f < NUM_FEATURES; f++)
		{
			fprintf(stdout, "%.1f%s", sample[f], (f == NUM_FEATURES - 1 ? "" : " "));
		}

		// > Seperator
		fprintf(stdout, "] -> [");

		// > Print our predictions
		for (f = 0; f < NUM_FEATURES - 1; f++)
		{
			fprintf(stdout, "%f%s", yValues[f], (f == NUM_FEATURES - 2 ? "" : " "));
		}
		fprintf(stdout, "]\n");
	}

	// > Find our largest value
	int cat = nn_maxindex(yValues);
	return cat;
}

// ------------------- //
// Mathmatic Functions //
// ------------------- //
static double nn_hypertan(double x)
{
	if (x < -20.0) return -1.0;
	else if (x > 20.0) return 1.0;
	else return tanh(x);
}

static double* nn_softmax(double* sums)
{
	// > Variables
	double sum = 0.0;
	int i;

	// > Clear Result
	for (i = 0; i < sizeof(sums); i++)
		res[i] = 0.0;

	// > Apply Softmax
	for (i = 0; i < sizeof(sums); i++)
	{
		sum += exp(sums[i]);
	}

	for (i = 0; i < sizeof(sums); i++)
	{
		res[i] = exp(sums[i]) / sum;
	}

	// > Return all of our data
	return res;
}

// ------------------- //
// Debugging Functions //
// ------------------- //
static void nn_showinfo()
{
	// > Debug
	if (isDebugging)
	{
		fprintf(stdout, "XX |   Showing information\n");
	}
}

static void nn_printnnmatrix()
{
	// > Variables
	int rows = NUM_TRAINING_SAMPLES, cols = NUM_FEATURES + numOutput;
	int r, c;

	// > Show matrix if we're debugging
	if (isDebugging)
	{
		fprintf(stdout, "-----------------------------------------------------------------------\n");
		for (r = 0; r < rows; r++)
		{
			fprintf(stdout, "[%d]: ", r);
			for (c = 0; c < cols; c++)
			{
				fprintf(stdout, "%.1f%s", myModelNN[r][c], (c == (cols - 1) ? "" : ", "));
			}
			fprintf(stdout, "\n");
		}
		fprintf(stdout, "-----------------------------------------------------------------------\n");
	}
}

static void nn_printprediction(int uniary, char lbl)
{
	// > Only print our prediction if we're debugging
	if (isDebugging && isShowingPredictions)
	{
		fprintf(stdout, "-> Prediction input: %d, label found with same uniary index: %c\n", uniary, lbl);
	}
}

static void nn_displayweights()
{
	// > Variables
	int i, w, t = 0;

	// > Debug
	if (isDebugging)
	{
		fprintf(stdout, "-----------------------------------------------------------------------\n");

		// > [IH]
		for (i = 0; i < numInput; i++)
		{
			for (w = 0; w < numHidden; w++)
			{
				fprintf(stdout, "IH |   Weight  [%s%d]  [%s]: |%s%f |\n", (w < 10 ? "0" : ""), w, (ihWeights[i][w] < 0 ? "-" : "+"), (ihWeights[i][w] > 0 ? " " : ""), ihWeights[i][w]);
				t++;
			}
		}

		// > [IH-B]
		for (i = 0; i < numHidden; i++)
		{
			fprintf(stdout, "HB |   Weight  [%s%d]  [%s]: |%s%f |\n", (i < 10 ? "0" : ""), i, (hBiases[i] < 0 ? "-" : "+"), (hBiases[i] > 0 ? " " : ""), hBiases[i]);
			t++;
		}

		// > [HO]
		for (i = 0; i < numHidden; i++)
		{
			for (w = 0; w < numOutput; w++)
			{
				fprintf(stdout, "HO |   Weight  [%s%d]  [%s]: |%s%f |\n", (w < 10 ? "0" : ""), w, (hoWeights[i][w] < 0 ? "-" : "+"), (hoWeights[i][w] > 0 ? " " : ""), hoWeights[i][w]);
				t++;
			}
		}

		// > [HO-B]
		for (i = 0; i < numOutput; i++)
		{
			fprintf(stdout, "OB |   Weight  [%s%d]  [%s]: |%s%f |\n", (i < 10 ? "0" : ""), i, (oBiases[i] < 0 ? "-" : "+"), (oBiases[i] > 0 ? " " : ""), oBiases[i]);
			t++;
		}

		fprintf(stdout, "-----------------------------------------------------------------------\n");
		fprintf(stdout, "XX |     Input Nodes: %d\n", numInput);
		fprintf(stdout, "XX |    Hidden Nodes: %d\n", numHidden);
		fprintf(stdout, "XX |    Output Nodes: %d\n", numOutput);
		fprintf(stdout, "XX |      Bias Nodes: %d\n", (numHidden + numOutput) / numOutput);
		fprintf(stdout, "-----------------------------------------------------------------------\n");
		fprintf(stdout, "XX |      IH Weights: %d\n", numInput*numHidden);
		fprintf(stdout, "XX |      HO Weights: %d\n", numHidden*numOutput);
		fprintf(stdout, "XX |    Bias Weights: %d\n", numHidden + numOutput);
		fprintf(stdout, "XX |   Total Weights: %d\n", t);
		fprintf(stdout, "-----------------------------------------------------------------------\n");
	}
}