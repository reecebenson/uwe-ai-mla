//
//  TrainAndTest.c
//  MLCoursework
//
//  This is a fairly inefficient implentation that does not use any dynamic memory allocation
// because that wouldnot be safe on the DEWIS marking system
//
//  Created by Jim Smith on 06/02/2017.
//  Copyright © 2017 Jim SmithJim Smith. All rights reserved.
//

#include "TrainAndTest.h"
#include <math.h>



//declare this array as static but make it available to any function in this file
//in case we want to store the training examples  and use them later
static    double myModel[NUM_TRAINING_SAMPLES][NUM_FEATURES];
static char myModelLabels[NUM_TRAINING_SAMPLES];
static int trainingSetSize = 0;

////////////////
//My variables//
////////////////

struct modelData {
	double data[NUM_FEATURES];
	int labelIndex;
};
struct Column {
	char labels;
	int index;
	struct modelData point[NUM_SAMPLES];
} static column[NUM_SAMPLES], column_copy[NUM_SAMPLES];


static double	outputNode[NUM_FEATURES * 2];
static double	output[NUM_FEATURES * 2];
static double	hiddenLayer[NUM_FEATURES * 2];
static double	hiddenLayerOutput[NUM_FEATURES * 2];
static double	w[NUM_SAMPLES];
static int		category;
static double	min;
static double	max;
static int		hiddenNodes;
static int		biasWeightCount;
static int		hiddenToOutput;
static double	bias;
static char classificationArray[NUM_SAMPLES];

double			fRand(double fMin, double fMax);

double sigmoid(double nodeWeights) {
	return 1 / (1 + exp(-nodeWeights));
}


double normlizeData(double currentValue, double maxCategoryValue, double minCategoryValue) {
	return (currentValue - minCategoryValue) / (maxCategoryValue - minCategoryValue);
}


double meanSqrError(double target, double output) {
	return ((target - output)*(target - output)) / 2;
}
double errorTotalOverOutputDerivitive(double target, double output) {
	return -(target - output);
}

double outputOVerNetDev(double output) {
	return (output * (1 - output));
}


int  train(double **trainingSamples, char *trainingLabels, int numSamples, int numFeatures)
{
	int returnval = 1;
	int sample, feature;
	int i = 0, j = 0, k = 0;
	double finalMeanCount = 0;


	//clean the model because C leaves whatever is in the memory
	for (sample = 0; sample < NUM_TRAINING_SAMPLES; sample++) {
		for (feature = 0; feature<NUM_FEATURES; feature++) {
			myModel[sample][feature] = 0.0;
		}
	}

	//sanity checking
	if (numFeatures > NUM_FEATURES || numSamples > NUM_TRAINING_SAMPLES) {
		//fprintf(stdout, "error: called train with data set larger than spaced allocated to store it");
		returnval = 0;
	}

	//make a simple copy of the data we are being passed but don't do anything with it
	//I'm just giving you this for the sake of people less familiar with pointers etc.


	if (returnval == 1) {
		//store the labels and the feature values
		trainingSetSize = numSamples;
		int index, feature;
		for (index = 0; index < numSamples; index++) {
			myModelLabels[index] = trainingLabels[index];
			for (feature = 0; feature < numFeatures; feature++) {
				myModel[index][feature] = trainingSamples[index][feature];
			}
		}
		//fprintf(stdout, "data stored locally \n");
	}//end else



	 /////////////////////////////////////////////
	 //Sort labels and data into label structure//
	 /////////////////////////////////////////////
	 ///////////////////////////////////////////////////////////////////////////////////////////////////
	 //Store myModelLabels into uniary array e.g a = [1,0,0],b = [0,1,0] and c = [0,0,1] for iris data//
	 ///////////////////////////////////////////////////////////////////////////////////////////////////



	int newLabel = 0;
	char labelmable;
	char classificationArray[NUM_SAMPLES];
	classificationArray[0] = myModelLabels[0];

	for (i = 0; i < numSamples; i++) {
		labelmable = myModelLabels[i];
		newLabel = 1;

		for (j = 0; j < category; j++) {
			if (labelmable == classificationArray[j]) {
				newLabel = 0;
			}
		}
		if (newLabel > 0) {
			classificationArray[category] = labelmable;
			category++;
		}
	}

	int outputArray[NUM_FEATURES * 2][NUM_FEATURES * 2];

	for (i = 0; i < category; i++) {
		for (k = 0; k < category; k++) {
			if (i == k) {
				outputArray[i][k] = 1;
			}
			else {
				outputArray[i][k] = 0;
			}
		}
	}
	for (k = 0; k < category; k++) {
		for (i = 0; i < numSamples; i++) {
			if (classificationArray[k] == myModelLabels[i])
			{
				for (j = 0; j < numFeatures; j++) {
					column[k].point[column[k].index].data[j] = myModel[i][j];
					column[k].point[column[k].index].labelIndex = i;
				}
				column[k].labels = classificationArray[k];
				column[k].index++;
			}
		}
	}


	//------------------------------------------------------------------------------------------------------------------------------------//

	//////////////////////////////////
	//Multilayer pereceptron attempt//
	//////////////////////////////////

	int m, n;
	double totalError = 0;

	hiddenNodes = floor((numFeatures + category) / 2);
	int weights = (numFeatures*hiddenNodes) + (hiddenNodes*category) + (hiddenNodes + category);
	bias = 1.0;
	double alpha = 0.24;
	int index = 0;


	//clear memory
	for (m = 0; m < hiddenNodes; m++) {
		hiddenLayer[m] = 0.0;
		hiddenLayerOutput[m] = 0.0;
	}
	for (m = 0; m < category; m++) {
		outputNode[m] = 0.0;
		output[m] = 0.0;
	}



	////////////////////////
	//normalize data 0 - 1//
	////////////////////////

	int l = 0;
	max = 0;
	min = 100;

	/*for (i = 0; i < category; i++) {
		for (j = 0; j < numFeatures; j++) {
			for (k = 0; k < column[i].index; k++) {
				if (column[i].point[k].data[j] > max) {
					max = column[i].point[k].data[j];
				}
				if (column[i].point[k].data[j] < min) {
					min = column[i].point[k].data[j];
				}

			}
			for (m = 0; m < column[i].index; m++) {
				column[i].point[m].data[j] = normlizeData(column[i].point[m].data[j], max, min);
			}
			for (n = 0; n < column[i].index; n++) {
			}
		}
	}*/


	/////////////////////////////////////////////
	//Set our weight values in our weight array// // use a value somewhere between 0.5 and -0.5
	/////////////////////////////////////////////

	for (i = 0; i < weights; i++) {
		w[i] = fRand(-0.1, 0.1);

		fprintf(stdout, "w[%d] = %f;\n", i, w[i]);
	}

	// Set Weights
	/*w[0] = -0.099750;
	w[1] = 0.012717;
	w[2] = -0.061339;
	w[3] = 0.061748;
	w[4] = 0.017002;
	w[5] = -0.004025;
	w[6] = -0.029942;
	w[7] = 0.079192;
	w[8] = 0.064568;
	w[9] = 0.049321;
	w[10] = -0.065178;
	w[11] = 0.071789;
	w[12] = 0.042100;
	w[13] = 0.002707;
	w[14] = -0.039201;
	w[15] = -0.097003;
	w[16] = -0.081719;
	w[17] = -0.027110;
	w[18] = -0.070537;
	w[19] = -0.066820;
	w[20] = 0.097705;
	w[21] = -0.010862;
	w[22] = -0.076183;
	w[23] = -0.099066;
	w[24] = -0.098218;
	w[25] = -0.024424;
	w[26] = 0.006333;*/

	/*for (i = 0; i < weights; i++) {
		w[i] = 0;
	}*/

	///////////////////////////////////////////////////////////////
	//Perform multilayer perceptron training on our data set here//
	///////////////////////////////////////////////////////////////

	int counter = 0;
	int p = 0;
	int q = 0;
	int epochs = 50000;
	int epochCount;
	double error[NUM_SAMPLES];

	index = 0;
	hiddenToOutput = 0;
	biasWeightCount = ((numFeatures*hiddenNodes) + (hiddenNodes * category));
	hiddenToOutput = (numFeatures * hiddenNodes);



	for (epochCount = 0; epochCount < epochs; epochCount++) {
		totalError = 0;

		for (k = 0; k < category; k++) {
			for (j = 0; j < column[k].index; j++) {


				////////////////
				//Feed Forward//
				////////////////

				////////////////////////
				//hidden layer sorting//
				////////////////////////

				for (n = 0; n < hiddenNodes; n++) {
					for (m = 0; m < numFeatures; m++) {
						hiddenLayer[n] += (w[counter] * column[k].point[j].data[m]);
						counter += hiddenNodes;
					}
				
					for (l = 0; l < hiddenNodes; l++) {
						hiddenLayer[n] += (bias * w[biasWeightCount]);
						biasWeightCount++;
					}
					biasWeightCount = ((numFeatures*hiddenNodes) + (hiddenNodes * category));

					index++;
					counter = 0;
					counter += index;


				}
	
				index = 0;
				counter = 0;
				for (p = 0; p < hiddenNodes; p++) {
					hiddenLayerOutput[p] = sigmoid(hiddenLayer[p]);

				}
				
				//end of hidden node sorting --------------------------------------------------------------------------------------------

				///////////////////
				//for each output//
				///////////////////

				for (l = 0; l < category; l++) {
					for (q = 0; q < hiddenNodes; q++) {
						outputNode[l] += (w[hiddenToOutput] * hiddenLayerOutput[q]); 
						hiddenToOutput += category;
					}
					
					index++;
					biasWeightCount += hiddenNodes;
					for (q = 0; q < category; q++) {
						outputNode[l] += (bias * w[biasWeightCount]);
						biasWeightCount++;
					}
					biasWeightCount = ((numFeatures*hiddenNodes) + (hiddenNodes * category));
					hiddenToOutput = ((numFeatures * hiddenNodes)) + index;
				}

				double sum = 0;
				for (l = 0; l < category; l++) {
					sum += exp(outputNode[l]);
					output[l] = exp(outputNode[l]);
				

				}
				for (l = 0; l < category; l++) {
					output[l] = output[l] / sum;
				}
				
			
				//end of output node sorting --------------------------------------------------------------------------------------------

				////////////////////////////////////////////////
				//calcualte error using squared error function//
				////////////////////////////////////////////////

				double target = 0;
				for (l = 0; l < category; l++) {
					if (outputArray[k][l] == 0) {
						target = 0.01;

					}
					else {
						target = 0.99;

					}

					error[l] = meanSqrError(target, output[l]);
				}
				for (l = 0; l < category; l++) {
					totalError += error[l];
				}

				// end of error --------------------------------------------------------------------------------------------------------------------------

				////////////////////
				//back propagation//
				////////////////////

				//////////////////////////
				//hidden to output layer//
				//////////////////////////

				hiddenToOutput = (numFeatures * hiddenNodes);
				double errorTotalDev[NUM_FEATURES * 2];
				double outNetDev[NUM_FEATURES * 2];
				double totalNetInputChange[NUM_FEATURES * 2];
				double devOfErrorTotalWeight[NUM_FEATURES * 2];
				double copyOfW[NUM_SAMPLES];
				index = 0;

				for (l = 0; l < category; l++) {
					if (outputArray[k][l] == 0) {
						target = 0.01;

					}
					else {
						target = 0.99;

					}
					errorTotalDev[l] = errorTotalOverOutputDerivitive(target, output[l]);
					outNetDev[l] = outputOVerNetDev(output[l]);
					for (q = 0; q < hiddenNodes; q++) {
						totalNetInputChange[q] = hiddenLayerOutput[q];
					}
					for (q = 0; q < hiddenNodes; q++) {
						devOfErrorTotalWeight[q] = errorTotalDev[l] * outNetDev[l] * totalNetInputChange[q];
					}
					for (q = 0; q < hiddenNodes; q++) {
						copyOfW[hiddenToOutput] = w[hiddenToOutput];
						w[hiddenToOutput] = w[hiddenToOutput] - (alpha * devOfErrorTotalWeight[q]);
						hiddenToOutput += category;
					}
					index++;
					hiddenToOutput = (numFeatures * hiddenNodes) + index;
					biasWeightCount += hiddenNodes;
					for (q = 0; q < category; q++) {
						w[biasWeightCount] = w[biasWeightCount] - (alpha * devOfErrorTotalWeight[q]);
						biasWeightCount++;
					}
					biasWeightCount = ((numFeatures*hiddenNodes) + (hiddenNodes * category));

				}
				hiddenToOutput = (numFeatures * hiddenNodes);

				/////////////////////////
				//input to hidden layer//
				/////////////////////////
				double outputErrorOverNet[NUM_FEATURES * 2];
				double outputErrorsOverHiddenOutputs[NUM_FEATURES * 2];
				double errorTotalOverHiddenOutput[NUM_FEATURES * 2];
				double hiddenSigOutDev[NUM_FEATURES * 2];
				double errorTotalOverWeights[NUM_FEATURES * 2];

				index = 0;
				int z;
				for (l = 0; l < hiddenNodes; l++) {
					for (q = 0; q < category; q++) {
						outputErrorOverNet[q] = errorTotalDev[q] * outNetDev[q];
					}
					for (q = 0; q < category; q++) {
						outputErrorsOverHiddenOutputs[q] = copyOfW[hiddenToOutput] * outputErrorOverNet[q];
						hiddenToOutput++;
					}
					for (q = 0; q < category; q++) {
						errorTotalOverHiddenOutput[q] = 0;
					}
					for (q = 0; q < category; q++) {
						errorTotalOverHiddenOutput[l] += outputErrorsOverHiddenOutputs[q];
					}
					for (q = 0; q < category; q++) {
						hiddenSigOutDev[q] = outputOVerNetDev(hiddenLayerOutput[q]);
					}
					for (z = 0; z < numFeatures; z++) {
						errorTotalOverWeights[z] = errorTotalOverHiddenOutput[l] * hiddenSigOutDev[l] * column[k].point[j].data[z];
						w[index] = w[index] - (alpha * errorTotalOverWeights[z]);
						index++;
					}
					for (q = 0; q < category; q++) {
						w[biasWeightCount] = w[biasWeightCount] - (alpha * errorTotalOverWeights[q]);
						biasWeightCount++;
					}
					biasWeightCount = ((numFeatures*hiddenNodes) + (hiddenNodes * category));

				}
				hiddenToOutput = (numFeatures * hiddenNodes);



				// ------------------------------------------------------------------------------------------------------------------------------------------
				for (l = 0; l < category; l++) {
					outputNode[l] = 0;
				}
				for (l = 0; l < category; l++) {
					hiddenLayer[l] = 0;
				}
				for (l = 0; l < category; l++) {
					error[l] = 0;
				}

				index = 0;
				counter = 0;
				hiddenToOutput = ((numFeatures * hiddenNodes));
				biasWeightCount = ((numFeatures*hiddenNodes) + (hiddenNodes * category));
			}
		}
			//printf("Total Error ---> %f\t\n", totalError/numSamples);
	}

	for (i = 0; i < weights; i++) {
		fprintf(stdout, "w[%d] = %f;\n", i, w[i]);
	}

	return returnval;
}


char  predictLabel(double *sample, int numFeatures)
{
	char prediction;
	int index = 0, j = 0, count = 0, predictionIndex = -1;
	int m, n, k, l, q, p;
	int counter = 0;
	double closest = 100;
	biasWeightCount = ((numFeatures*hiddenNodes) + (hiddenNodes * category));
	hiddenToOutput = ((numFeatures * hiddenNodes));

	for (l = 0; l < category; l++) {
		outputNode[l] = 0;
	}
	for (l = 0; l < category; l++) {
		hiddenLayer[l] = 0;
	}



	/////////////
	//normalize//
	/////////////
	/*for (j = 0; j < numFeatures; j++) {
		sample[j] = (sample[j] - min) / (max - min);
	}*/

	////////////////////////
	//hidden layer sorting//
	////////////////////////
	for (n = 0; n < hiddenNodes; n++) {
		for (m = 0; m < numFeatures; m++) {
			hiddenLayer[n] += (w[counter] * sample[m]);
			counter += 3;
		}
		for (l = 0; l < (hiddenNodes); l++) {
			hiddenLayer[n] += (bias * w[biasWeightCount]);
			biasWeightCount++;
		}
		biasWeightCount = ((numFeatures*hiddenNodes) + (hiddenNodes * category));

		index++;
		counter = 0;
		counter += index;


	}
	index = 0;
	counter = 0;
	for (p = 0; p < hiddenNodes; p++) {
		hiddenLayerOutput[p] = sigmoid(hiddenLayer[p]);

	}

	//end of hidden node sorting --------------------------------------------------------------------------------------------

	///////////////////
	//for each output//
	///////////////////

	for (l = 0; l < category; l++) {
		for (q = 0; q < hiddenNodes; q++) {
			outputNode[l] += (w[hiddenToOutput] * hiddenLayerOutput[q]);
			hiddenToOutput += 3;
		}
		
		index++;
		biasWeightCount += 3;
		for (q = 0; q < category; q++) {
			outputNode[l] += (bias * w[biasWeightCount]);
			biasWeightCount++;
		}
		biasWeightCount = ((numFeatures*hiddenNodes) + (hiddenNodes * category));
		hiddenToOutput = ((numFeatures * hiddenNodes)) + index;
	}
	double sum = 0;
	
	for (l = 0; l < category; l++) {
		sum += exp(outputNode[l]);
		output[l] = exp(outputNode[l]);
	

	}
	for (l = 0; l < category; l++) {
		output[l] = output[l] / sum;
	}
		printf("Output --> %.2f\t %.2f\t %.2f\t \n", output[0], output[1], output[2]);

	double best = 0;
	int predictCounter = 0;
	for (l = 0; l < category; l++) {
		if (output[l] > 0 && output[l] > best) {
			best = output[l];
			predictCounter = l;
		}
	}


	prediction = column[predictCounter].labels;


	return prediction;

}


double fRand(double fMin, double fMax)
{
	double f = (double)rand() / 2147483647;//32767;// 2147483647;
	return fMin + f * (fMax - fMin);
}