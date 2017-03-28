//
//  exampleMain.c
//  MLCoursework
//
//  Created by Jim Smith on 23/02/2017.
//  Copyright © 2017 Jim SmithJim Smith. All rights reserved.
//

#include <stdio.h>

#include "MLCoursework.h"
#include "TrainAndTest.h"


//this is just to show how your programme might be called
//you should not assume that any particular data set is being used to test your code
//BUT you can assume that NUM_TRAINING_SAMPLES, NUM_FEATURES and NUM_TEST_SAMPLES are defined



#include "IrisData.h"

#include <stdlib.h>




int main(int argc, const char * argv[])
{
    
    int sample, trainingsamples=0,testsamples=0,feature=0;
    int ok=1;
    int correct=0, wrong=0;
    char prediction;
    int returnval=0;
    double   mySample[NUM_FEATURES];
    
    //allocate space for data storage
    double **trainingSet = calloc(NUM_TRAINING_SAMPLES , sizeof(double *));
    for(sample=0;sample < NUM_TRAINING_SAMPLES; sample++)
        trainingSet[sample] = calloc(NUM_FEATURES, sizeof(double));
    
                                  
    char *trainingLabels = calloc(NUM_TRAINING_SAMPLES , sizeof(char));
    
    double **testSet = calloc(NUM_TEST_SAMPLES , sizeof(double *));
    for(sample=0;sample < NUM_TEST_SAMPLES; sample++)
        testSet[sample] = calloc(NUM_FEATURES, sizeof(double));

    char * testLabels = calloc(NUM_TEST_SAMPLES, sizeof(char));
    

    
    //simple 2/3: 1/3 split of iris data into training and test set matrices
    
    for(sample=0; sample< IRIS_SET_SIZE;  sample++)
      {
        if(sample%3==0)
          {
            for (feature=0;feature < IRISFEATURES;  feature++)
              testSet[testsamples][feature] = iris_data[sample][feature];

            testLabels[testsamples] =iris_labels[sample];
            testsamples++;
          }
        else
          {
            for (feature=0;feature < IRISFEATURES;  feature++)
              trainingSet[trainingsamples][feature] = iris_data[sample][feature];
            
            trainingLabels[trainingsamples] = iris_labels[sample];
            trainingsamples++;
          }
      }
    
    
    
    //call the train function
    ok= train(  &trainingSet[0], trainingLabels ,trainingsamples, NUM_FEATURES);
    if( ok!= 1)
      {
        printf("there was a problem running the train() function\n");
        returnval=0;
      }
    
    else
      {
        //print the results from the training set for information
        printf("On the training set:\n");
        for(sample=0; sample < trainingsamples; sample++)
          {
            
            // Make a copy of the sample ot be classified because it is faster to access a local copy
            //and pass a pointer to it, and also  I don't want to overwrite it by accident
            for(feature = 0 ; feature < NUM_FEATURES; feature++)
                mySample[feature] = trainingSet[sample][feature];
            
            prediction = predictLabel( mySample, NUM_FEATURES);
            if (prediction == trainingLabels[sample])
                correct++;
			else
			{
				//printf("[FAIL]  -> Looking for '%c', received '%c'. [%i] -> [%.1f,%.1f,%.1f,%.1f]\n", trainingLabels[sample], prediction, sample, mySample[0], mySample[1], mySample[2], mySample[3]);
				wrong++;
			}
          }
        
        printf(" %d correct %d incorrect, accuracy = %f %%\n", correct,wrong, 100.0*(float)correct/(correct+wrong));
        

        //now the results that matter - from the test set
        printf("On the test set:\n");        correct=wrong=0;
        for(sample=0; sample < testsamples; sample++)
          {
            //copy into temp array ot pass into fiunction
            for(feature = 0 ; feature < NUM_FEATURES; feature++)
                mySample[feature] = testSet[sample][feature];
            
            prediction = predictLabel(mySample, NUM_FEATURES);
			if (prediction == testLabels[sample])
			{
				correct++;
			}
            else
            {
				//printf("[FAIL]  -> Looking for '%c', received '%c'. [%i] -> [%.1f,%.1f,%.1f,%1.f]\n", testLabels[sample], prediction, sample, mySample[0], mySample[1], mySample[2], mySample[3]);
                wrong++;
            }
          }
        
        printf(" %d correct %d incorrect, accuracy = %f %%\n", correct,wrong, 100.0*(float)correct/(correct+wrong));
        
      }
    
	getchar();
    return returnval;
    
}